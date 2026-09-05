#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "testing.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace autoparser;

static std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open file: " + path.string());
    }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

static common_chat_template load_glm53_template() {
    std::filesystem::path root;
    if (const char * workspace = std::getenv("GITHUB_WORKSPACE")) {
        root = workspace;
    } else {
        root = std::filesystem::path(__FILE__).parent_path().parent_path();
    }
    return common_chat_template(read_text_file(root / "models/templates/GLM-5.3-Flash.jinja"), "", "");
}

static json build_tools() {
    json properties = json::object();
    properties["command"] = json::object({ { "type", "string" } });
    properties["description"] = json::object({ { "type", "string" } });
    properties["workdir"] = json::object({ { "type", "string" } });

    json parameters = json::object();
    parameters["type"] = "object";
    parameters["properties"] = properties;
    parameters["required"] = json::array({ "command" });

    return json::array({
        json::object({
            { "type", "function" },
            { "function", json::object({
                { "name", "exec_shell_command" },
                { "description", "Execute a shell command" },
                { "parameters", parameters },
            }) },
        }),
    });
}

static common_chat_msg parse(testing & t, const common_peg_arena & parser, const std::string & label,
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

static void assert_call(testing & t, const std::string & label, const common_chat_msg & msg,
                        const std::string & command) {
    if (!t.assert_equal(label + ": one call", size_t(1), msg.tool_calls.size())) {
        return;
    }
    t.assert_equal(label + ": name", std::string("exec_shell_command"), msg.tool_calls[0].name);
    try {
        auto args = json::parse(msg.tool_calls[0].arguments);
        t.assert_equal(label + ": command", command, args.at("command").get<std::string>());
    } catch (const std::exception & e) {
        t.assert_true(label + ": valid args JSON: " + std::string(e.what()), false);
    }
}

static common_peg_arena build_tool_only_parser(const common_chat_template & tmpl, const generation_params & inputs,
                                                autoparser::autoparser & analysis) {
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

static void test_dialects(testing & t) {
    auto tmpl = load_glm53_template();
    generation_params inputs;
    inputs.tools = build_tools();
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.parallel_tool_calls = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    inputs.enable_thinking = true;

    autoparser::autoparser analysis;
    auto parser = build_tool_only_parser(tmpl, inputs, analysis);
    t.assert_equal("format", tool_format::TAG_WITH_TAGGED, analysis.tools.format.mode);

    const std::string command = "cd \"L:/AI_pictures_generate/Manual LLAMA_CPP\" && ls -la | head -40";
    const std::string canonical =
        "<tool_call>exec_shell_command"
        "<arg_key>command</arg_key>"
        "<arg_value>cd \"L:/AI_pictures_generate/Manual LLAMA_CPP\" && ls -la | head -40</arg_value>"
        "</tool_call>";
    const std::string name_json =
        "<tool_call>exec_shell_command{\"command\":\"cd \\\"L:/AI_pictures_generate/Manual LLAMA_CPP\\\" && ls -la | head -40\"}</tool_call>";
    const std::string flat_json =
        "<tool_call>{\"function-name\":\"exec_shell_command\",\"command\":\"cd \\\"L:/AI_pictures_generate/Manual LLAMA_CPP\\\" && ls -la | head -40\"}</tool_call>";

    assert_call(t, "canonical", parse(t, parser, "canonical", canonical), command);
    assert_call(t, "name+json", parse(t, parser, "name+json", name_json), command);
    assert_call(t, "flat-json", parse(t, parser, "flat-json", flat_json), command);
}

static void test_streaming(testing & t) {
    auto tmpl = load_glm53_template();
    generation_params inputs;
    inputs.tools = build_tools();
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.parallel_tool_calls = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    inputs.enable_thinking = true;

    autoparser::autoparser analysis;
    auto parser = build_tool_only_parser(tmpl, inputs, analysis);

    const std::string a = "<tool_call>exec_shell_command{\"command\":\"pwd\"}</tool_call>";
    const std::string b = "<tool_call>{\"function-name\":\"exec_shell_command\",\"command\":\"pwd\"}</tool_call>";
    for (const auto & sample : { a, b }) {
        for (size_t i = 1; i <= sample.size(); ++i) {
            common_peg_parse_context ctx(sample.substr(0, i), COMMON_PEG_PARSE_FLAG_LENIENT);
            auto result = parser.parse(ctx);
            if (!result.success()) {
                continue;
            }
            common_chat_msg msg;
            common_chat_peg_mapper mapper(msg);
            mapper.from_ast(ctx.ast, result);
            if (sample.substr(0, i).find("<tool_call>") != std::string::npos) {
                t.assert_true("tool marker never leaks into content", msg.content.find("<tool_call>") == std::string::npos);
            }
        }
    }
}

static void test_reasoning_boundary(testing & t) {
    auto tmpl = load_glm53_template();
    generation_params inputs;
    inputs.tools = build_tools();
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.parallel_tool_calls = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs.enable_thinking = true;

    autoparser::autoparser analysis;
    analysis.analyze_template(tmpl);
    auto parser = analysis.build_parser(inputs, "");
    auto msg = parse(t, parser, "reasoning boundary",
        "<think>Need to inspect the repository.<tool_call>exec_shell_command{\"command\":\"pwd\"}</tool_call>");
    t.assert_equal("reasoning", std::string("Need to inspect the repository."), msg.reasoning_content);
    assert_call(t, "reasoning boundary", msg, "pwd");
}

static void test_none(testing & t) {
    auto tmpl = load_glm53_template();
    generation_params inputs;
    inputs.messages = json::array({ json::object({ { "role", "user" }, { "content", "Hello" } }) });
    inputs.add_generation_prompt = true;
    inputs.extra_context["reasoning_effort"] = "none";
    const std::string rendered = common_chat_template_direct_apply(tmpl, inputs);
    t.assert_true("no Reasoning Effort: None", rendered.find("Reasoning Effort: None") == std::string::npos);
    const std::string suffix = "<|assistant|><think></think>";
    t.assert_true("none closes think", rendered.size() >= suffix.size() && rendered.rfind(suffix) == rendered.size() - suffix.size());
}

int main() {
    testing t(std::cout);
    t.verbose = true;
    t.test("glm53 dialects", test_dialects);
    t.test("glm53 streaming", test_streaming);
    t.test("glm53 reasoning boundary", test_reasoning_boundary);
    t.test("glm53 reasoning none", test_none);
    return t.summary();
}
