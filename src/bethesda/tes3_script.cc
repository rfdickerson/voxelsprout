#include "bethesda/tes3_script.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cctype>
#include <limits>
#include <sstream>

namespace odai::bethesda {
namespace {

std::string trim(std::string value) {
    const auto whitespace = [](unsigned char ch) { return std::isspace(ch) != 0; };
    while (!value.empty() && whitespace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && whitespace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::string stripComment(std::string line) {
    bool quote = false;
    for (std::size_t i = 0u; i < line.size(); ++i) {
        if (line[i] == '"') quote = !quote;
        if (line[i] == ';' && !quote) {
            line.resize(i);
            break;
        }
    }
    return trim(std::move(line));
}

std::vector<std::string> splitArguments(std::string_view text) {
    std::vector<std::string> result;
    std::string token;
    bool quote = false;
    int parentheses = 0;
    const auto flush = [&]() {
        token = trim(std::move(token));
        if (!token.empty()) result.push_back(std::move(token));
        token.clear();
    };
    for (const char ch : text) {
        if (ch == '"') quote = !quote;
        if (!quote && ch == '(') ++parentheses;
        if (!quote && ch == ')') --parentheses;
        if (!quote && parentheses == 0 && (std::isspace(static_cast<unsigned char>(ch)) || ch == ',')) {
            flush();
        } else {
            token.push_back(ch);
        }
    }
    flush();
    return result;
}

std::pair<std::string, std::string> firstToken(std::string_view text) {
    std::size_t begin = 0u;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    std::size_t end = begin;
    while (end < text.size() && !std::isspace(static_cast<unsigned char>(text[end])) &&
           text[end] != ',') ++end;
    std::string first(text.substr(begin, end - begin));
    while (end < text.size() &&
           (std::isspace(static_cast<unsigned char>(text[end])) || text[end] == ',')) ++end;
    return {first, std::string(text.substr(end))};
}

struct ParsedCallLine {
    std::string target;
    std::string command;
    std::string arguments;
};

ParsedCallLine parseCallLine(std::string_view line) {
    ParsedCallLine result;
    bool quote = false;
    std::size_t arrow = std::string_view::npos;
    for (std::size_t index = 0u; index + 1u < line.size(); ++index) {
        if (line[index] == '"') quote = !quote;
        if (!quote && line[index] == '-' && line[index + 1u] == '>') {
            arrow = index;
            break;
        }
    }
    std::string_view commandLine = line;
    if (arrow != std::string_view::npos) {
        result.target = trim(std::string(line.substr(0u, arrow)));
        commandLine = line.substr(arrow + 2u);
    }
    std::size_t begin = 0u;
    while (begin < commandLine.size() &&
           (std::isspace(static_cast<unsigned char>(commandLine[begin])) ||
            commandLine[begin] == ',')) ++begin;
    std::size_t end = begin;
    while (end < commandLine.size()) {
        const unsigned char ch = static_cast<unsigned char>(commandLine[end]);
        if (!std::isalnum(ch) && commandLine[end] != '_') break;
        ++end;
    }
    result.command = normalizeTes3Symbol(commandLine.substr(begin, end - begin));
    while (end < commandLine.size() &&
           (std::isspace(static_cast<unsigned char>(commandLine[end])) ||
            commandLine[end] == ',' || commandLine[end] == ':')) ++end;
    result.arguments = std::string(commandLine.substr(end));
    return result;
}

std::string removeOuterParens(std::string value) {
    value = trim(std::move(value));
    if (value.size() >= 2u && value.front() == '(' && value.back() == ')') {
        value = trim(value.substr(1u, value.size() - 2u));
    }
    return value;
}

bool startsWithKeyword(std::string_view line, std::string_view keyword) {
    if (line.size() < keyword.size()) return false;
    if (normalizeTes3Symbol(line.substr(0u, keyword.size())) != keyword) return false;
    return line.size() == keyword.size() ||
        std::isspace(static_cast<unsigned char>(line[keyword.size()])) != 0 ||
        line[keyword.size()] == '(';
}

std::uint64_t hashSource(std::string_view source) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const unsigned char ch : source) {
        hash ^= ch;
        hash *= 1099511628211ull;
    }
    return hash;
}

struct ExpressionToken {
    enum class Kind : std::uint8_t { Number, String, Identifier, Operator, Left, Right };
    Kind kind = Kind::Identifier;
    std::string text;
};

std::vector<ExpressionToken> tokenizeExpression(std::string_view expression) {
    std::vector<ExpressionToken> result;
    std::size_t i = 0u;
    while (i < expression.size()) {
        const unsigned char ch = static_cast<unsigned char>(expression[i]);
        if (std::isspace(ch) || ch == ',') { ++i; continue; }
        if (ch == '"') {
            std::string value;
            ++i;
            while (i < expression.size() && expression[i] != '"') value.push_back(expression[i++]);
            if (i < expression.size()) ++i;
            result.push_back({ExpressionToken::Kind::String, std::move(value)});
            continue;
        }
        if (ch == '(' || ch == ')') {
            result.push_back({ch == '(' ? ExpressionToken::Kind::Left :
                ExpressionToken::Kind::Right, std::string(1u, static_cast<char>(ch))});
            ++i;
            continue;
        }
        if (std::isdigit(ch) || (ch == '.' && i + 1u < expression.size() &&
            std::isdigit(static_cast<unsigned char>(expression[i + 1u])))) {
            const std::size_t begin = i++;
            while (i < expression.size() &&
                   (std::isalnum(static_cast<unsigned char>(expression[i])) ||
                    expression[i] == '.' || expression[i] == '+' || expression[i] == '-')) {
                if ((expression[i] == '+' || expression[i] == '-') &&
                    expression[i - 1u] != 'e' && expression[i - 1u] != 'E') break;
                ++i;
            }
            result.push_back({ExpressionToken::Kind::Number,
                std::string(expression.substr(begin, i - begin))});
            continue;
        }
        if (ch == '=' || ch == '!' || ch == '<' || ch == '>' || ch == '+' ||
            ch == '-' || ch == '*' || ch == '/') {
            std::string op(1u, static_cast<char>(ch));
            ++i;
            if (i < expression.size() &&
                (expression[i] == '=' || (op == "-" && expression[i] == '>'))) {
                op.push_back(expression[i++]);
            }
            result.push_back({ExpressionToken::Kind::Operator, std::move(op)});
            continue;
        }
        const std::size_t begin = i++;
        while (i < expression.size()) {
            const unsigned char next = static_cast<unsigned char>(expression[i]);
            if (std::isspace(next) || expression[i] == ',' || expression[i] == '(' ||
                expression[i] == ')' || expression[i] == '=' || expression[i] == '!' ||
                expression[i] == '<' || expression[i] == '>' || expression[i] == '+' ||
                expression[i] == '-' || expression[i] == '*' || expression[i] == '/') break;
            ++i;
        }
        std::string value(expression.substr(begin, i - begin));
        const std::string normalized = normalizeTes3Symbol(value);
        if (normalized == "and" || normalized == "or" || normalized == "not") {
            result.push_back({ExpressionToken::Kind::Operator, normalized});
        } else {
            result.push_back({ExpressionToken::Kind::Identifier, std::move(value)});
        }
    }
    return result;
}

double numeric(const Tes3Value& value) {
    if (value.type == Tes3ValueType::Number) return value.number;
    if (value.type == Tes3ValueType::String) {
        double result = 0.0;
        const auto parsed = std::from_chars(
            value.string.data(), value.string.data() + value.string.size(), result);
        if (parsed.ec == std::errc{}) return result;
    }
    return value.truthy() ? 1.0 : 0.0;
}

bool looksLikeFunction(std::string_view symbol) {
    const std::string name = normalizeTes3Symbol(symbol);
    // PC-prefixed identifiers in shipped content are overwhelmingly globals
    // (PCVampire, PCWerewolf, quest hand-off flags, and similar). Treat only
    // the one expression-form PC opcode as a native; statement-form PC calls
    // are discovered from their Call instruction normally.
    return name.starts_with("get") || name == "pcexpelled" ||
        name == "menumode" || name == "cellchanged" ||
        name == "scriptrunning" || name == "random" || name == "getdistance" ||
        name == "getinterior" || name == "getlineofsight";
}

std::optional<double> parseNumber(std::string_view text) {
    double value = 0.0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size()) return value;
    return std::nullopt;
}

}  // namespace

std::string normalizeTes3Symbol(std::string_view symbol) {
    std::string result(symbol);
    for (char& ch : result) {
        if (static_cast<unsigned char>(ch) < 0x80u) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
    }
    return result;
}

Tes3Value Tes3Value::fromNumber(double value) {
    Tes3Value result;
    result.type = Tes3ValueType::Number;
    result.number = value;
    return result;
}

Tes3Value Tes3Value::fromString(std::string value) {
    Tes3Value result;
    result.type = Tes3ValueType::String;
    result.string = std::move(value);
    return result;
}

Tes3Value Tes3Value::fromObject(ObjectId value) {
    Tes3Value result;
    result.type = Tes3ValueType::Object;
    result.object = std::move(value);
    return result;
}

bool Tes3Value::truthy() const {
    if (type == Tes3ValueType::Number) return number != 0.0 && !std::isnan(number);
    if (type == Tes3ValueType::String) return !string.empty();
    if (type == Tes3ValueType::Object) return object.valid();
    return false;
}

bool Tes3CompileResult::success() const {
    return std::none_of(diagnostics.begin(), diagnostics.end(),
        [](const Tes3CompileDiagnostic& item) { return item.error; });
}

Tes3CompileResult Tes3ScriptCompiler::compile(
    std::string_view source, std::string scriptId) const {
    Tes3CompileResult result;
    result.program.id = normalizeTes3Symbol(scriptId);
    result.program.sourceHash = hashSource(source);
    enum class BlockKind : std::uint8_t { If, While };
    struct Block {
        BlockKind kind = BlockKind::If;
        std::size_t branch = 0u;
        std::size_t loop = 0u;
        std::vector<std::size_t> endJumps;
        bool sawElse = false;
        std::uint32_t line = 0u;
    };
    std::vector<Block> blocks;
    std::istringstream input{std::string(source)};
    std::string line;
    std::uint32_t lineNumber = 0u;
    bool sawBegin = false;
    while (std::getline(input, line)) {
        ++lineNumber;
        line = stripComment(std::move(line));
        if (line.empty()) continue;
        const auto [headOriginal, tailOriginal] = firstToken(line);
        const std::string head = normalizeTes3Symbol(headOriginal);
        if (head == "begin") {
            if (sawBegin) result.diagnostics.push_back({lineNumber, true, "duplicate begin"});
            sawBegin = true;
            if (result.program.id.empty()) result.program.id = normalizeTes3Symbol(firstToken(tailOriginal).first);
            continue;
        }
        if (head == "end") {
            if (!blocks.empty()) result.diagnostics.push_back({lineNumber, true, "end before block is closed"});
            Tes3Instruction instruction;
            instruction.op = Tes3OpCode::Return;
            instruction.sourceLine = lineNumber;
            result.program.instructions.push_back(std::move(instruction));
            continue;
        }
        if (head == "short" || head == "long" || head == "float" || head == "ref") {
            const std::string local = normalizeTes3Symbol(firstToken(tailOriginal).first);
            if (local.empty()) {
                result.diagnostics.push_back({lineNumber, true, head + " declaration has no name"});
            } else if (!result.program.locals.emplace(local,
                    head == "short" ? Tes3LocalType::Short :
                    head == "long" ? Tes3LocalType::Long :
                    head == "float" ? Tes3LocalType::Float : Tes3LocalType::Reference).second) {
                result.diagnostics.push_back({lineNumber, true, "duplicate local " + local});
            }
            continue;
        }
        if (head == "set") {
            bool quote = false;
            std::size_t toPosition = std::string::npos;
            for (std::size_t index = 0u; index + 1u < tailOriginal.size(); ++index) {
                if (tailOriginal[index] == '"') quote = !quote;
                if (quote || normalizeTes3Symbol(tailOriginal.substr(index, 2u)) != "to") continue;
                const bool left = index == 0u ||
                    std::isspace(static_cast<unsigned char>(tailOriginal[index - 1u])) != 0;
                const bool right = index + 2u == tailOriginal.size() ||
                    std::isspace(static_cast<unsigned char>(tailOriginal[index + 2u])) != 0;
                if (left && right) { toPosition = index; break; }
            }
            const std::string destination = toPosition == std::string::npos
                ? std::string{} : trim(tailOriginal.substr(0u, toPosition));
            const std::string expression = toPosition == std::string::npos
                ? std::string{} : trim(tailOriginal.substr(toPosition + 2u));
            if (destination.empty() || expression.empty()) {
                result.diagnostics.push_back({lineNumber, true,
                    "set requires '<name> to <expression>': " + line});
                continue;
            }
            Tes3Instruction instruction;
            instruction.op = Tes3OpCode::Assign;
            instruction.sourceLine = lineNumber;
            instruction.destination = normalizeTes3Symbol(destination);
            instruction.expression = expression;
            result.program.instructions.push_back(std::move(instruction));
            continue;
        }
        if (startsWithKeyword(line, "if") || startsWithKeyword(line, "elseif")) {
            const bool elseIf = startsWithKeyword(line, "elseif");
            if (elseIf) {
                if (blocks.empty() || blocks.back().kind != BlockKind::If || blocks.back().sawElse) {
                    result.diagnostics.push_back({lineNumber, true, "elseif without matching if"});
                    continue;
                }
                Tes3Instruction jump;
                jump.op = Tes3OpCode::Jump;
                jump.sourceLine = lineNumber;
                blocks.back().endJumps.push_back(result.program.instructions.size());
                result.program.instructions.push_back(std::move(jump));
                result.program.instructions[blocks.back().branch].jump = result.program.instructions.size();
            }
            const std::size_t keywordLength = elseIf ? 6u : 2u;
            Tes3Instruction branch;
            branch.op = Tes3OpCode::BranchIfFalse;
            branch.sourceLine = lineNumber;
            branch.expression = removeOuterParens(line.substr(keywordLength));
            const std::size_t branchIndex = result.program.instructions.size();
            result.program.instructions.push_back(std::move(branch));
            if (elseIf) blocks.back().branch = branchIndex;
            else blocks.push_back(Block{BlockKind::If, branchIndex, 0u, {}, false, lineNumber});
            continue;
        }
        if (head == "else") {
            if (blocks.empty() || blocks.back().kind != BlockKind::If || blocks.back().sawElse) {
                result.diagnostics.push_back({lineNumber, true, "else without matching if"});
                continue;
            }
            Tes3Instruction jump;
            jump.op = Tes3OpCode::Jump;
            jump.sourceLine = lineNumber;
            blocks.back().endJumps.push_back(result.program.instructions.size());
            result.program.instructions.push_back(std::move(jump));
            result.program.instructions[blocks.back().branch].jump = result.program.instructions.size();
            blocks.back().sawElse = true;
            continue;
        }
        if (head == "endif") {
            if (blocks.empty() || blocks.back().kind != BlockKind::If) {
                result.diagnostics.push_back({lineNumber, true, "endif without matching if"});
                continue;
            }
            Block block = std::move(blocks.back());
            blocks.pop_back();
            const std::size_t end = result.program.instructions.size();
            if (!block.sawElse) result.program.instructions[block.branch].jump = end;
            for (const std::size_t jump : block.endJumps) result.program.instructions[jump].jump = end;
            continue;
        }
        if (head == "while") {
            Tes3Instruction branch;
            branch.op = Tes3OpCode::BranchIfFalse;
            branch.sourceLine = lineNumber;
            branch.expression = removeOuterParens(line.substr(5u));
            const std::size_t index = result.program.instructions.size();
            result.program.instructions.push_back(std::move(branch));
            blocks.push_back(Block{BlockKind::While, index, index, {}, false, lineNumber});
            continue;
        }
        if (head == "endwhile") {
            if (blocks.empty() || blocks.back().kind != BlockKind::While) {
                result.diagnostics.push_back({lineNumber, true, "endwhile without matching while"});
                continue;
            }
            const Block block = blocks.back();
            blocks.pop_back();
            Tes3Instruction jump;
            jump.op = Tes3OpCode::Jump;
            jump.sourceLine = lineNumber;
            jump.jump = block.loop;
            result.program.instructions.push_back(std::move(jump));
            result.program.instructions[block.branch].jump = result.program.instructions.size();
            continue;
        }
        if (head == "return") {
            Tes3Instruction instruction;
            instruction.op = Tes3OpCode::Return;
            instruction.sourceLine = lineNumber;
            result.program.instructions.push_back(std::move(instruction));
            continue;
        }

        Tes3Instruction call;
        call.op = Tes3OpCode::Call;
        call.sourceLine = lineNumber;
        const ParsedCallLine parsedCall = parseCallLine(line);
        call.target = parsedCall.target;
        call.command = parsedCall.command;
        if (call.command.empty()) {
            result.diagnostics.push_back({lineNumber, true, "command name is empty"});
            continue;
        }
        call.arguments = splitArguments(parsedCall.arguments);
        result.program.commands.insert(call.command);
        result.program.instructions.push_back(std::move(call));
    }
    for (const Block& block : blocks) {
        result.diagnostics.push_back({block.line, true,
            block.kind == BlockKind::If ? "unterminated if" : "unterminated while"});
    }
    if (result.program.instructions.empty() ||
        result.program.instructions.back().op != Tes3OpCode::Return) {
        Tes3Instruction instruction;
        instruction.op = Tes3OpCode::Return;
        instruction.sourceLine = lineNumber + 1u;
        result.program.instructions.push_back(std::move(instruction));
    }
    // Native functions can appear inside assignments and branch expressions
    // rather than as standalone statement calls. Include them in the closure
    // report so strict content checks cannot silently miss gameplay queries.
    for (const Tes3Instruction& instruction : result.program.instructions) {
        if (instruction.expression.empty()) continue;
        for (const ExpressionToken& token : tokenizeExpression(instruction.expression)) {
            const std::string symbol = normalizeTes3Symbol(token.text);
            if (token.kind == ExpressionToken::Kind::Identifier &&
                !result.program.locals.contains(symbol) && looksLikeFunction(symbol)) {
                result.program.commands.insert(symbol);
            }
        }
    }
    if (result.program.id.empty()) result.program.id = "dialogue_result";
    return result;
}

void Tes3NativeRegistry::registerNative(Tes3NativeDefinition definition) {
    definition.name = normalizeTes3Symbol(definition.name);
    m_definitions.insert_or_assign(definition.name, std::move(definition));
}

const Tes3NativeDefinition* Tes3NativeRegistry::find(std::string_view command) const {
    const auto found = m_definitions.find(normalizeTes3Symbol(command));
    return found == m_definitions.end() ? nullptr : &found->second;
}

Tes3NativeRegistry Tes3NativeRegistry::coreRuntimeRegistry() {
    Tes3NativeRegistry result;
    constexpr std::string_view implemented[] = {
        "journal", "setjournalindex", "getjournalindex", "addtopic", "choice", "goodbye",
        "additem", "removeitem", "getitemcount", "enable", "disable", "delete",
        "startscript", "stopscript", "scriptrunning", "placeatpc", "gethealth",
        "sethealth", "modcurrenthealth", "getdisabled", "random",
        "getdisposition", "setdisposition", "moddisposition", "getpcrank",
        "pcjoinfaction", "pcraiserank", "pclowerrank", "pcexpell", "pcexpelled",
        "getreputation", "setreputation", "modreputation", "modpcfacrep",
        "setpcfacrep", "getdeadcount", "menumode", "cellchanged",
        "getsecondspassed", "getpos", "setpos", "getangle", "setangle",
        "getscale", "setscale", "modscale", "position", "positioncell",
        "aitravel", "aiwander", "aifollow", "aifollowcell", "aiescort",
        "aiescortcell", "getaipackagedone", "getcurrentaipackage",
        "startcombat", "stopcombat", "getfight", "setfight", "modfight",
        "messagebox", "showmap", "setpccrimelevel", "modpccrimelevel",
        "payfine", "payfinethief", "getdistance", "getpccell", "getinterior",
        "getfatigue", "setfatigue", "modfatigue", "modcurrentfatigue",
        "getmagicka", "setmagicka", "modmagicka", "modcurrentmagicka",
        "getalarm", "setalarm", "modalarm", "getflee", "setflee", "modflee",
        "gethello", "sethello", "modhello", "lock", "unlock", "getlocked",
        "activate", "placeatme", "addspell", "removespell", "getspell",
        "equip", "forcerun", "clearforcerun", "getforcerun", "getattacked",
        "getsoundplaying", "getlevel", "getrace", "gethealthgetratio",
        "getpccrimelevel", "getpcinjail", "forcegreeting",
        "disableplayercontrols", "enableplayercontrols",
        "disableplayerfighting", "enableplayerfighting",
        "disableplayerjumping", "enableplayerjumping",
        "disableplayermagic", "enableplayermagic",
        "disableplayerviewswitch", "enableplayerviewswitch",
        "disableteleporting", "enableteleporting",
        "disablelevitation", "enablelevitation",
        "disablevanitymode", "enablevanitymode", "enablerest",
        "getsquareroot", "getcurrenttime", "getcurrentweather", "changeweather",
        "getpcjumping", "getpcrunning", "getpcsleep", "getpcsneaking",
        "getpctraveling", "getbuttonpressed", "getspellreadied",
        "getweapondrawn", "getwerewolfkills", "gotojail", "wakeuppc",
        "onactivate", "getmoving", "getwaterlevel", "setwaterlevel",
        "modwaterlevel", "getstartingangle", "setatstart", "pcclearexpelled",
        "raiserank", "lowerrank", "modfactionreaction", "gettarget", "setdelete",
        "placeitem", "placeitemcell", "drop", "move", "moveworld", "rotate",
        "rotateworld", "face", "modhealth", "resurrect", "forcejump",
        "clearforcejump", "forcemovejump", "clearforcemovejump",
        "getforcemovejump", "forcesneak", "clearforcesneak", "getforcesneak",
        "cast", "getspelleffects"};
    for (const std::string_view name : implemented) {
        result.registerNative({std::string(name), Tes3NativeDisposition::Implemented, true});
    }
    constexpr std::string_view actorStats[] = {
        "strength", "intelligence", "willpower", "agility", "speed", "endurance",
        "personality", "luck", "block", "armorer", "mediumarmor", "heavyarmor",
        "bluntweapon", "longblade", "axe", "spear", "athletics", "enchant",
        "destruction", "alteration", "illusion", "conjuration", "mysticism",
        "restoration", "alchemy", "unarmored", "security", "sneak", "acrobatics",
        "lightarmor", "shortblade", "marksman", "mercantile", "speechcraft",
        "handtohand"};
    for (const std::string_view stat : actorStats) {
        for (const std::string_view operation : {"get", "set", "mod"}) {
            result.registerNative({std::string(operation) + std::string(stat),
                Tes3NativeDisposition::Implemented, true});
        }
    }
    constexpr std::string_view presentation[] = {
        "playsound", "playsound3d", "playsoundvp", "playsound3dvp",
        "playloopsound3d", "playloopsound3dvp", "stopsound", "streammusic",
        "say", "saydone", "playgroup", "loopgroup", "skipanim", "fadein",
        "fadeout", "togglemenus", "title", "playbink"};
    for (const std::string_view name : presentation) {
        result.registerNative({std::string(name), Tes3NativeDisposition::PresentationOnly, false});
    }
    return result;
}

bool Tes3ScriptVm::registerProgram(Tes3ScriptProgram program, std::string& outError) {
    program.id = normalizeTes3Symbol(program.id);
    if (program.id.empty() || program.instructions.empty()) {
        outError = "MWScript program requires an id and instructions";
        return false;
    }
    m_programs.insert_or_assign(program.id, std::move(program));
    outError.clear();
    return true;
}

std::uint64_t Tes3ScriptVm::start(
    std::string_view program, ObjectId owner, std::string& outError) {
    const std::string id = normalizeTes3Symbol(program);
    const auto found = m_programs.find(id);
    if (found == m_programs.end()) {
        outError = "unknown MWScript program " + id;
        return 0u;
    }
    Tes3ScriptThread thread;
    thread.id = m_nextThreadId++;
    thread.program = id;
    thread.owner = std::move(owner);
    for (const auto& [name, type] : found->second.locals) {
        (void)type;
        thread.locals.emplace(name, Tes3Value::fromNumber(0.0));
    }
    m_threads.emplace(thread.id, std::move(thread));
    outError.clear();
    return m_nextThreadId - 1u;
}

Tes3Value Tes3ScriptVm::lookup(
    std::string_view name, const Tes3ScriptThread& thread) const {
    const std::string normalized = normalizeTes3Symbol(name);
    const auto local = thread.locals.find(normalized);
    if (local != thread.locals.end()) return local->second;
    const auto event = thread.eventVariables.find(normalized);
    if (event != thread.eventVariables.end()) return event->second;
    const auto global = m_globals.find(normalized);
    if (global != m_globals.end()) return global->second;
    return {};
}

std::optional<Tes3Value> Tes3ScriptVm::evaluate(
    std::string_view expression, Tes3ScriptThread& thread, std::uint64_t tick,
    const Tes3NativeExecutor& execute, std::string& outError) const {
    const std::vector<ExpressionToken> tokens = tokenizeExpression(expression);
    class Parser {
    public:
        Parser(const std::vector<ExpressionToken>& tokens, const Tes3ScriptVm& vm,
               Tes3ScriptThread& thread, std::uint64_t tick,
               const Tes3NativeExecutor& execute, std::string& error)
            : m_tokens(tokens), m_vm(vm), m_thread(thread), m_tick(tick),
              m_execute(execute), m_error(error) {}

        Tes3Value parse() { return parseOr(); }
        bool complete() const { return m_position == m_tokens.size(); }

    private:
        bool accept(std::string_view op) {
            if (m_position >= m_tokens.size() ||
                m_tokens[m_position].kind != ExpressionToken::Kind::Operator ||
                normalizeTes3Symbol(m_tokens[m_position].text) != op) return false;
            ++m_position;
            return true;
        }
        Tes3Value parseOr() {
            Tes3Value left = parseAnd();
            while (accept("or")) left = Tes3Value::fromNumber(left.truthy() || parseAnd().truthy());
            return left;
        }
        Tes3Value parseAnd() {
            Tes3Value left = parseCompare();
            while (accept("and")) left = Tes3Value::fromNumber(left.truthy() && parseCompare().truthy());
            return left;
        }
        Tes3Value parseCompare() {
            Tes3Value left = parseAdd();
            if (m_position >= m_tokens.size() ||
                m_tokens[m_position].kind != ExpressionToken::Kind::Operator) return left;
            const std::string op = m_tokens[m_position].text;
            if (op != "=" && op != "==" && op != "!=" && op != ">" && op != ">=" &&
                op != "<" && op != "<=") return left;
            ++m_position;
            const Tes3Value right = parseAdd();
            bool result = false;
            if ((left.type == Tes3ValueType::String || right.type == Tes3ValueType::String) &&
                (op == "=" || op == "==" || op == "!=")) {
                const std::string a = left.type == Tes3ValueType::String ? left.string : std::to_string(numeric(left));
                const std::string b = right.type == Tes3ValueType::String ? right.string : std::to_string(numeric(right));
                result = normalizeTes3Symbol(a) == normalizeTes3Symbol(b);
                if (op == "!=") result = !result;
            } else {
                const double a = numeric(left);
                const double b = numeric(right);
                if (op == "=" || op == "==") result = a == b;
                else if (op == "!=") result = a != b;
                else if (op == ">") result = a > b;
                else if (op == ">=") result = a >= b;
                else if (op == "<") result = a < b;
                else result = a <= b;
            }
            return Tes3Value::fromNumber(result ? 1.0 : 0.0);
        }
        Tes3Value parseAdd() {
            Tes3Value left = parseMultiply();
            while (m_position < m_tokens.size() &&
                   m_tokens[m_position].kind == ExpressionToken::Kind::Operator &&
                   (m_tokens[m_position].text == "+" || m_tokens[m_position].text == "-")) {
                const bool add = m_tokens[m_position++].text == "+";
                const double right = numeric(parseMultiply());
                left = Tes3Value::fromNumber(numeric(left) + (add ? right : -right));
            }
            return left;
        }
        Tes3Value parseMultiply() {
            Tes3Value left = parseUnary();
            while (m_position < m_tokens.size() &&
                   m_tokens[m_position].kind == ExpressionToken::Kind::Operator &&
                   (m_tokens[m_position].text == "*" || m_tokens[m_position].text == "/")) {
                const bool multiply = m_tokens[m_position++].text == "*";
                const double right = numeric(parseUnary());
                left = Tes3Value::fromNumber(multiply ? numeric(left) * right :
                    (right == 0.0 ? 0.0 : numeric(left) / right));
            }
            return left;
        }
        Tes3Value parseUnary() {
            if (accept("not")) return Tes3Value::fromNumber(!parseUnary().truthy());
            if (accept("-")) return Tes3Value::fromNumber(-numeric(parseUnary()));
            return parsePrimary();
        }
        Tes3Value parsePrimary() {
            if (m_position >= m_tokens.size()) { m_error = "missing expression operand"; return {}; }
            const ExpressionToken token = m_tokens[m_position++];
            if (token.kind == ExpressionToken::Kind::Left) {
                Tes3Value value = parseOr();
                if (m_position >= m_tokens.size() ||
                    m_tokens[m_position].kind != ExpressionToken::Kind::Right) {
                    m_error = "unclosed expression parenthesis";
                    return {};
                }
                ++m_position;
                return value;
            }
            if (token.kind == ExpressionToken::Kind::Number) {
                return Tes3Value::fromNumber(parseNumber(token.text).value_or(0.0));
            }
            if (token.kind == ExpressionToken::Kind::String &&
                !(m_position < m_tokens.size() &&
                  m_tokens[m_position].kind == ExpressionToken::Kind::Operator &&
                  m_tokens[m_position].text == "->")) {
                return Tes3Value::fromString(token.text);
            }
            if (token.kind != ExpressionToken::Kind::Identifier) {
                if (token.kind != ExpressionToken::Kind::String) {
                    m_error = "unexpected expression token " + token.text;
                    return {};
                }
            }
            std::string target;
            std::string function = token.text;
            if (m_position < m_tokens.size() &&
                m_tokens[m_position].kind == ExpressionToken::Kind::Operator &&
                m_tokens[m_position].text == "->") {
                target = token.kind == ExpressionToken::Kind::String
                    ? '"' + token.text + '"' : token.text;
                ++m_position;
                if (m_position >= m_tokens.size() ||
                    m_tokens[m_position].kind != ExpressionToken::Kind::Identifier) {
                    m_error = "target-qualified expression has no function";
                    return {};
                }
                function = m_tokens[m_position++].text;
            }
            const Tes3Value known = target.empty() ? m_vm.lookup(function, m_thread) : Tes3Value{};
            if (target.empty() &&
                (known.type != Tes3ValueType::None || !looksLikeFunction(function))) return known;
            Tes3NativeCall call;
            call.target = std::move(target);
            call.command = normalizeTes3Symbol(function);
            call.tick = m_tick;
            call.owner = m_thread.owner;
            while (m_position < m_tokens.size() && call.arguments.size() < 4u) {
                const ExpressionToken& next = m_tokens[m_position];
                if (next.kind == ExpressionToken::Kind::Operator ||
                    next.kind == ExpressionToken::Kind::Right) break;
                ++m_position;
                if (next.kind == ExpressionToken::Kind::Number) {
                    call.arguments.push_back(Tes3Value::fromNumber(parseNumber(next.text).value_or(0.0)));
                } else if (next.kind == ExpressionToken::Kind::String) {
                    call.arguments.push_back(Tes3Value::fromString(next.text));
                } else {
                    const Tes3Value argument = m_vm.lookup(next.text, m_thread);
                    call.arguments.push_back(argument.type == Tes3ValueType::None
                        ? Tes3Value::fromString(next.text) : argument);
                }
            }
            const Tes3NativeResult result = m_execute(call);
            if (!result.error.empty()) m_error = result.error;
            return result.value;
        }

        const std::vector<ExpressionToken>& m_tokens;
        const Tes3ScriptVm& m_vm;
        Tes3ScriptThread& m_thread;
        std::uint64_t m_tick;
        const Tes3NativeExecutor& m_execute;
        std::string& m_error;
        std::size_t m_position = 0u;
    };
    if (tokens.empty()) { outError = "empty expression"; return std::nullopt; }
    Parser parser(tokens, *this, thread, tick, execute, outError);
    Tes3Value value = parser.parse();
    if (!outError.empty()) return std::nullopt;
    if (!parser.complete()) { outError = "trailing expression tokens"; return std::nullopt; }
    return value;
}

Tes3VmStepResult Tes3ScriptVm::step(
    std::uint64_t tick, std::uint32_t instructionBudget,
    const Tes3NativeExecutor& execute) {
    Tes3VmStepResult result;
    for (auto& [threadId, thread] : m_threads) {
        (void)threadId;
        if (thread.state != Tes3ThreadState::Running) continue;
        const auto program = m_programs.find(thread.program);
        if (program == m_programs.end()) {
            thread.state = Tes3ThreadState::Failed;
            thread.error = "registered program disappeared";
            result.diagnostics.push_back(thread.error);
            continue;
        }
        while (thread.state == Tes3ThreadState::Running &&
               thread.instruction < program->second.instructions.size()) {
            if (result.instructions >= instructionBudget) {
                result.diagnostics.push_back("MWScript instruction budget exhausted at " +
                    thread.program + ":" + std::to_string(thread.instruction));
                return result;
            }
            const Tes3Instruction& instruction =
                program->second.instructions[thread.instruction];
            ++result.instructions;
            if (instruction.op == Tes3OpCode::Return) {
                thread.state = Tes3ThreadState::Completed;
                ++result.completedThreads;
                break;
            }
            if (instruction.op == Tes3OpCode::Jump) {
                thread.instruction = instruction.jump;
                continue;
            }
            if (instruction.op == Tes3OpCode::Assign ||
                instruction.op == Tes3OpCode::BranchIfFalse) {
                std::string error;
                const std::optional<Tes3Value> value = evaluate(
                    instruction.expression, thread, tick, execute, error);
                if (!value.has_value()) {
                    thread.state = Tes3ThreadState::Failed;
                    thread.error = "line " + std::to_string(instruction.sourceLine) + ": " + error;
                    result.diagnostics.push_back(thread.program + ": " + thread.error);
                    break;
                }
                if (instruction.op == Tes3OpCode::Assign) {
                    const std::string destination = normalizeTes3Symbol(instruction.destination);
                    if (thread.locals.contains(destination) || thread.eventVariables.contains(destination)) {
                        thread.locals[destination] = *value;
                    } else {
                        m_globals[destination] = *value;
                    }
                    ++thread.instruction;
                } else {
                    thread.instruction = value->truthy() ? thread.instruction + 1u : instruction.jump;
                }
                continue;
            }
            Tes3NativeCall call;
            call.target = instruction.target;
            call.command = instruction.command;
            call.tick = tick;
            call.owner = thread.owner;
            for (const std::string& argument : instruction.arguments) {
                if (argument.size() >= 2u && argument.front() == '"' && argument.back() == '"') {
                    call.arguments.push_back(Tes3Value::fromString(argument.substr(1u, argument.size() - 2u)));
                } else if (const std::optional<double> value = parseNumber(argument)) {
                    call.arguments.push_back(Tes3Value::fromNumber(*value));
                } else {
                    const Tes3Value resolved = lookup(argument, thread);
                    call.arguments.push_back(resolved.type == Tes3ValueType::None
                        ? Tes3Value::fromString(argument) : resolved);
                }
            }
            const Tes3NativeResult native = execute(call);
            if (!native.error.empty()) {
                thread.state = Tes3ThreadState::Failed;
                thread.error = "line " + std::to_string(instruction.sourceLine) + ": " + native.error;
                result.diagnostics.push_back(thread.program + ": " + thread.error);
                break;
            }
            ++thread.instruction;
            if (native.suspend) {
                thread.state = Tes3ThreadState::Suspended;
                thread.suspensionReason = native.suspensionReason;
            }
        }
        if (thread.state == Tes3ThreadState::Running &&
            thread.instruction >= program->second.instructions.size()) {
            thread.state = Tes3ThreadState::Completed;
            ++result.completedThreads;
        }
    }
    return result;
}

bool Tes3ScriptVm::resume(std::uint64_t threadId, std::string& outError) {
    const auto found = m_threads.find(threadId);
    if (found == m_threads.end() || found->second.state != Tes3ThreadState::Suspended) {
        outError = "MWScript thread is not suspended";
        return false;
    }
    found->second.state = Tes3ThreadState::Running;
    found->second.suspensionReason.clear();
    outError.clear();
    return true;
}

void Tes3ScriptVm::clear() {
    m_programs.clear();
    m_threads.clear();
    m_globals.clear();
    m_nextThreadId = 1u;
}

}  // namespace odai::bethesda
