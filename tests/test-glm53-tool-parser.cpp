#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "testing.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace autoparser;

static std::string read_text_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

static common_chat_template load_glm53_template() {
    return common_chat_template(
        read_text_file("models/templates/GLM-5.3-Flash.jinja"),
        "",
        "");
}

static json build_exec_shell_tools() {
    json properties = json::object();
    properties["command"] = json::object({ { "type", "string" } });
    properties["description"] = json::object({ { "type", "string" } });
    properties["workdir"] = json::object({ { "type", "string" } });

    json parameters = json::object();
    parameters["type"] = "object";
    parameters["properties"] = properties;
    parameters["required"] = json::array({ "command" });

    json function = json::object();
    function["name"] = "exec_shell_command";
    function["description"] = "Execute a shell command";
    function["parameters"] = parameters;

    return json::array({
        json::object({
            { "type", "function" },
            { "function", function },
        }),
    });
}

static common_peg_arena build_glm53_tool_parser(const common_chat_template & tmpl, const generation_params & inputs,
                                                 autoparser & analysis) {
    analysis.analyze_template(tmpl);
    return build_chat_peg_parser([&](common_chat_peg_builder & p) {
        parser_build_context ctx(p, inputs);
        ctx.reasoning_parser = p.eps();
        ctx.extracting_reasoning = false;
        ctx.reasoning = &analysis.reasoning;
        ctx.content = &analysis.content;
        return analysis.tools.build_parser(ctx);
    });
}

static common_chat_msg parse_complete(testing & t, const common_peg_arena & parser, const std::string & label,
                                      const std::string & input) {
    common_peg_parse_context ctx(input);
    auto result = parser.parse(ctx);
    if (!t.assert_true(label + ": parse success", result.success())) {
        return {};
    }

    common_chat_msg msg;
    common_chat_peg_mapper mapper(msg);
    mapper.from_ast(ctx.ast, result);
    return msg;
}

static void assert_exec_call(testing & t, const std::string & label, const common_chat_msg & msg,
                             const std::string & expected_command) {
    if (!t.assert_equal(label + ": one tool call", size_t(1), msg.tool_calls.size())) {
        return;
    }

    t.assert_equal(label + ": function name", std::string("exec_shell_command"), msg.tool_calls[0].name);

    try {
        auto args = json::parse(msg.tool_calls[0].arguments);
        t.assert_equal(label + ": command", expected_command, args.at("command").get<std::string>());
    } catch (const std::exception & e) {
        t.assert_true(label + ": arguments must be valid JSON: " + std::string(e.what()), false);
    }
}

static void test_tool_dialects(testing & t) {
    auto tmpl = load_glm53_template();

    generation_params inputs;
    inputs.tools = build_exec_shell_tools();
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.parallel_tool_calls = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    inputs.enable_thinking = true;

    autoparser analysis;
    auto parser = build_glm53_tool_parser(tmpl, inputs, analysis);

    t.assert_equal("GLM 5.3 template is analyzed as TAG_WITH_TAGGED",
                   tool_format::TAG_WITH_TAGGED, analysis.tools.format.mode);

    const std::string command = "cd \"L:/AI_pictures_generate/Manual LLAMA_CPP\" && ls -la | head -40";

    const std::string canonical =
        "<tool_call>exec_shell_command"
        "<arg_key>command</arg_key>"
        "<arg_value>cd \"L:/AI_pictures_generate/Manual LLAMA_CPP\" && ls -la | head -40</arg_value>"
        "</tool_call>";

    const std::string tag_with_json =
        R"(<tool_call>exec_shell_command{"command":"cd \"L:/AI_pictures_generate/Manual LLAMA_CPP\" && ls -la | head -40"}</tool_call>)";

    const std::string flat_json =
        R"(<tool_call>{"function-name":"exec_shell_command","command":"cd \"L:/AI_pictures_generate/Manual LLAMA_CPP\" && ls -la | head -40"}</tool_call>)";

    assert_exec_call(t, "canonical tagged", parse_complete(t, parser, "canonical tagged", canonical), command);
    assert_exec_call(t, "name plus JSON", parse_complete(t, parser, "name plus JSON", tag_with_json), command);
    assert_exec_call(t, "flat JSON", parse_complete(t, parser, "flat JSON", flat_json), command);
}

static void test_streaming_prefixes_do_not_become_content(testing & t) {
    auto tmpl = load_glm53_template();

    generation_params inputs;
    inputs.tools = build_exec_shell_tools();
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.parallel_tool_calls = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    inputs.enable_thinking = true;

    autoparser analysis;
    auto parser = build_glm53_tool_parser(tmpl, inputs, analysis);

    const std::string full = R"(<tool_call>exec_shell_command{"command":"pwd"}</tool_call>)";
    const std::string flat = R"(<tool_call>{"function-name":"exec_shell_command","command":"pwd"}</tool_call>)";

    for (const auto & sample : { full, flat }) {
        for (size_t i = 1; i <= sample.size(); ++i) {
            common_peg_parse_context ctx(sample.substr(0, i), COMMON_PEG_PARSE_FLAG_LENIENT);
            auto result = parser.parse(ctx);
            if (!result.success()) {
                continue;
            }

            common_chat_msg msg;
            common_chat_peg_mapper mapper(msg);
            mapper.from_ast(ctx.ast, result);

            // Once a real GLM tool-call marker has been recognized, it must never be published
            // as ordinary assistant content and then retracted on the next chunk. That retraction
            // is what used to trip common_chat_msg_diff::compute_diffs with "Invalid diff".
            if (sample.substr(0, i).find("<tool_call>") != std::string::npos) {
                t.assert_true("recognized tool prefix must not leak <tool_call> into content",
                              msg.content.find("<tool_call>") == std::string::npos);
            }
        }
    }
}

static void test_reasoning_can_end_at_tool_call(testing & t) {
    auto tmpl = load_glm53_template();

    generation_params inputs;
    inputs.tools = build_exec_shell_tools();
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.parallel_tool_calls = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs.enable_thinking = true;

    autoparser analysis;
    analysis.analyze_template(tmpl);
    auto parser = analysis.build_parser(inputs, "");

    const std::string input =
        "<think>Need to inspect the repository."
        "<tool_call>exec_shell_command{\"command\":\"pwd\"}</tool_call>";

    auto msg = parse_complete(t, parser, "implicit reasoning boundary", input);
    t.assert_equal("reasoning before tool call", std::string("Need to inspect the repository."), msg.reasoning_content);
    assert_exec_call(t, "implicit reasoning boundary", msg, "pwd");
}

static void test_reasoning_effort_none(testing & t) {
    auto tmpl = load_glm53_template();

    generation_params inputs;
    inputs.messages = json::array({
        json::object({ { "role", "user" }, { "content", "Hello" } }),
    });
    inputs.add_generation_prompt = true;
    inputs.extra_context["reasoning_effort"] = "none";

    const std::string rendered = common_chat_template_direct_apply(tmpl, inputs);

    t.assert_true("none does not emit Reasoning Effort system line",
                  rendered.find("Reasoning Effort: None") == std::string::npos);
    t.assert_true("none closes thinking in generation prompt",
                  rendered.size() >= std::string("<|assistant|><think></think>").size() &&
                  rendered.rfind("<|assistant|><think></think>") ==
                      rendered.size() - std::string("<|assistant|><think></think>").size());
}

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("glm53.tool_dialects", test_tool_dialects);
    t.test("glm53.streaming_prefixes", test_streaming_prefixes_do_not_become_content);
    t.test("glm53.reasoning_tool_boundary", test_reasoning_can_end_at_tool_call);
    t.test("glm53.reasoning_effort_none", test_reasoning_effort_none);

    return t.summary();
}
