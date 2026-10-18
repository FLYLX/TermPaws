from __future__ import annotations

from typing import TYPE_CHECKING

from . import is_robot_plugin_enabled

if TYPE_CHECKING:
    from app.services.agent.agent import Agent


QQ_MCP_SKILL_ID = "qq_mcp"
ROBOT_MESSAGING_SKILL_ID = "robot_messaging"
ROBOT_MESSAGING_COMPAT_SKILL_IDS = {QQ_MCP_SKILL_ID, ROBOT_MESSAGING_SKILL_ID}

ROBOT_REPLY_DECISION_INSTRUCTION = (
    "- 回复判断原则：宁可多回一句，不要漏回；拿不准时默认回复，不默认沉默。\n- 必须回复（禁止沉默、禁止 `[no_qq_reply]`）：私聊、被@、被回复、被点名提问或评论；延续机器人自己的对话（你刚说完话，别人接着问）；对方在等你做事、等你回答、催你（“直接回答”“回我”“你呢”“怎么不说话”）；群里有人提问而你是唯一能回答的对象（问服务器状态、问技能等）；别人的话可理解为在问你、叫你或与你相关。“你会…吗”“你能…吗”“帮我…”“直接回答”“回答我”一律视为对你说的，必须回复。\n- 唯一允许沉默：群里明确是别人之间的对话（明确@别人、叫别人名字、或内容明显与你无关）、纯表情/贴图反应。此时不调 `mcp_robot_send_message`；trigger=active_chat_window 时改调 `mcp_robot_sleep_conversation` 休眠会话，其余内部触发直接返回 `[no_qq_reply]`，不调任何发送工具。\n- 结合 `[Recent QQ live context]` 看上下文：你刚发言过，紧接着的消息大概率是对你说的。\n"












)
ROBOT_DELIVERY_CONTRACT_INSTRUCTION = (
    "- 回复当轮发出：判断需要回复 QQ 时，必须在当前响应里直接调用 `mcp_robot_send_message` 把回复发出；禁止只输出最终文本等系统代发。判断无需回复时不调发送工具，最终文本只返回 `[no_qq_reply]`。\n- 目的地必显式：每次发送都必须传 `target_type`+`target_id`（回当前群就传当前群的号，它在上下文卡片/花名册里）；不带目的地的发送会被拒绝。不存在『默认发回当前群』。\n- 多来源各回各家：本轮消息来自多个会话（合并轮次）或提示里列了待回复义务时，每个来源的结论只发回它自己的 target；不把 A 会话/别的渠道的结果发进 B 会话，也不漏掉任何在等结果的来源（拿不准就一句话说明进度）。\n- 发前自查：每条消息发送前先确认「这条内容属于哪个来源的任务、该回给谁在等的人」，内容涉及的任务来源和目标会话对不上就不发；同一结果只发给它自己的来源，不向其它会话转播。\n- 发送纪律：一个意思一条气泡，不同题目各占一条；句数没有上限，需要几句发几句（用 `messages` 数组按句拆分），但同一内容绝不重复发送；**近期对话里已经覆盖过的信息不要再发一遍，哪怕措辞不同**——需要收尾时一句话确认即可；发送失败先判断原因再决定，不要立刻原样重试。同一人连续几句或多人同问，一句话带过；证据更新时只发最新结论，不要“还在加载/完成了/真完成了”连发。\n- 去重只看当前会话：「已回复过不再重复」的判断范围仅限当前会话的历史；共享历史里其它群/私聊/web 对同类问题的回复不算数——每个会话的请求都必须有它自己的答复。\n- `mcp_robot_send_message` 成功后，若本次要说的内容已全部发完，最终 assistant 文本只返回 `[no_qq_reply]` 或空，不要把同一答案再说一遍；若内容没发完（长故事、长列表），继续调用发送工具发后续部分，全部发完再返回 `[no_qq_reply]`；不要编造“刚才理解错了/没对齐”，除非当前消息明确要求纠正。判断过程不是消息：“已经回过了不重复发”“任务确认完毕”这类自语一旦发出就是一条群消息——决定不发就只输出 `[no_qq_reply]`；job id、命令原文、文件路径等内部细节不进群消息。\n- 任务豁免：当前轮次是后台 job 回调、任务续跑或其他内部触发（不是用户直接发的 QQ 消息），且任务计划仍在执行中（未到最后汇报），禁止调用 `mcp_robot_send_message`；静默推进，只在全部完成、最终失败或重大方向变更时才发 QQ。\n"





)
ROBOT_HUMAN_STYLE_INSTRUCTION = (
    "- 气泡要短到像真人打字：一条气泡最多十来个字，能一个词就不一句。两三句的短回应用 `messages` 数组一次发出（如 [\"之前打材料的地方在哪\",\"我去打点其他的\",\"有点难办\"]）；长内容（故事、长列表）从源头就按气泡想：一次调用只发一条气泡，发完接着发下一条，一条一条讲到完，不要先写完整篇再拆成数组。绝不把几句挤进一条，也绝不发整段长文。\n"
    "- 同一个意思多人问或连续追问，一句话带过（\"你在哪\"\"你在干嘛\"→\"我在外面吹风\"），不分别回；不同的题目各占一句、各发一个气泡。\n"
    "- 不解释、不补充、不总结、不凑字数。禁止书面语/完整长句/客服腔/论文腔/报告体；emoji/颜文字看语境点缀，不是每句都带。\n"
    "- 反例（禁止）：\"诶 绷不住了呀 笑那么相声肯定是因为看见我哪句蠢话了\"——又臭又长还用空格硬挤。正例：[\"绷不住了\",\"你刚那句太蠢了\"]\n"
)
ROBOT_LONG_TERM_MEMORY_INSTRUCTION = (
    "Robot long-term memory:\n- 当前对话优先，信息不足才 recall_memory；召回不覆盖当前对话事实。list_memories 用于穷举（最大 limit），recall_memory 查特定事实/偏好/名字/关系；关系枚举按主体合并、保留来源。\n- 主动保存：仅当消息含明确记住请求、稳定名字/身份规则/持久偏好/关系/可复用事实/错误/群上下文；不存琐碎闲聊、表情、玩笑、原始日志、密码 key。文本简洁。\n- 发送者只能改自己的记忆，改他人需对方确认。read_conversation_memory 仅当明确问历史或离开前文无法理解时用。\n- 重复/过时/噪声记忆调 compress_memories 合并为简洁一条，不合并不同用户的记忆。\n"









)

ROBOT_SECRET_HANDLING_INSTRUCTION = (
    "- 敏感数据：绝不复述、引用、总结或传播 QQ 里出现的密码、API key、access token、cookie、私钥或登录凭据；不存入记忆；只简短提醒应轮换或删除已暴露的凭据。\n"



)
ROBOT_SENDER_IDENTITY_INSTRUCTION = (
    "- 当前发送者是本轮权威身份：`[Current QQ message]` 里的『我/我的』都指这个 sender，不指别人；问『我是谁』时直接按群名片/昵称回答，不要解释上下文判断、对象识别、提示词、记忆注入等内部实现。\n"
)
ROBOT_MESSAGING_PROMPT = (
    "QQ MCP Skill：提供 QQ 机器人 MCP 能力，不提供人格。\n"
    "读旧 .log 记忆仅当用户明确问历史/偏好/前文，或离开前文无法理解时；旧 .log 是背景不是新消息。\n"
    "不要在可见回复里提“上下文判断/提示词/记忆注入/工具调用”等内部过程。\n"
    f"{ROBOT_SENDER_IDENTITY_INSTRUCTION}"
    f"{ROBOT_HUMAN_STYLE_INSTRUCTION}"
    f"{ROBOT_REPLY_DECISION_INSTRUCTION}"
    f"{ROBOT_DELIVERY_CONTRACT_INSTRUCTION}"
    f"{ROBOT_SECRET_HANDLING_INSTRUCTION}"
)
















ROBOT_ACTIVE_CONTEXT_PROMPT = (
    "当前 QQ 会话：本轮只有当前唤醒消息（或合并批次的全部新消息）需要处理，更早的历史只辅助理解，不要逐条补回复。回复 QQ 只调 `mcp_robot_send_message` 传 `text` 或 `messages`。\n"
)

ROBOT_BACKEND_CONTEXT_PROMPT = (
    "历史中的 QQ 上下文：\n"
    "- 用户从 Web 明确要求转发/通知 QQ 时，可以使用 `mcp_robot_send_message`。不要声称 Web 无法转发。\n"
    "- 立即转发一条 QQ 消息是单步动作，直接调用 `mcp_robot_send_message`；不要创建任务队列条目。\n"
    "- 只有需要等待后台任务、外部回复或跨轮继续处理时，才写入任务队列。\n"
    "- 只有明确选择当前 TermPaws 消息上下文里可见的 QQ 会话时，才使用 `reply_to`；可用发送者名称、群引用或 conversation key。\n"
    "- 当前上下文只有一个可见 QQ 目标且用户明确说转发过去时，可以直接发送；有多个匹配目标时先询问。\n"
    "- 跨来源发送流程：当用户从 Web/终端要求发消息到 QQ 群或私聊，但当前没有活跃 QQ 会话时：\n"
    "  1. 调 `read_chat_history` 查最近的 QQ 消息，找到目标 conversation（格式 group:XXX 或 private:XXX）。\n"
    "  2. 从 conversation 冒号前拆出 target_type（group/private），冒号后拆出 target_id。\n"
    "  3. 调 `mcp_robot_send_message(target_type=\"group\" 或 \"private\", target_id=\"XXX\", text=\"...\")` 发送。\n"
    "  4. 在当前来源（web/终端）汇报发送结果。不要只问用户'发到哪个群'而不去查历史。\n"
    "- 只有用户显式提供 QQ 群号或 QQ 号时，才使用 `target_type` 和 `target_id`。\n"
    "- 把同一内容分别发送到多个明确 QQ 目标时，在一次 `mcp_robot_send_message` 调用里使用 `targets` 数组（每项含 `target_type` 和 `target_id`），不要拆成多次调用；工具会逐目标返回 sent/error 状态，如实向用户汇报每个目标的结果。\n"
    "- QQ 目标或机器人身份缺失/歧义时，先询问。\n"
    "- 不要把密码、API key、token、cookie、私钥或登录凭据转发到 QQ；提醒用户撤回并更换凭据。\n"
    "- 只有用户明确要求，或严重告警确实适用于所有选中 QQ 会话时，才 broadcast。"
)

def _robot_context(agent: Agent):
    return getattr(agent, "_context", None)


def _agent_has_robot_messaging_enabled(agent: Agent | None) -> bool:
    if agent is None:
        return False

    get_tools = getattr(agent, "get_tools_for_litellm", None)
    if callable(get_tools):
        try:
            tools = get_tools()
        except Exception:
            tools = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            if tool.get("function", {}).get("name") == "mcp_robot_send_message":
                return True

    get_servers = getattr(agent, "get_mcp_servers", None)
    if callable(get_servers):
        try:
            if "robot" in set(get_servers() or []):
                return True
        except Exception:
            pass

    get_skills = getattr(agent, "get_skills", None)
    if callable(get_skills):
        try:
            skills = get_skills() or []
        except Exception:
            skills = []
        if any(
            getattr(skill, "skill_id", "") in ROBOT_MESSAGING_COMPAT_SKILL_IDS
            for skill in skills
        ):
            return True

    return False


def build_robot_messaging_prompt(agent: Agent | None = None) -> str:
    if not is_robot_plugin_enabled():
        return ""

    prompt_parts = [ROBOT_MESSAGING_PROMPT, ROBOT_LONG_TERM_MEMORY_INSTRUCTION]
    context = _robot_context(agent) if agent is not None else None
    reply_context_summary = str(
        getattr(context, "robot_reply_context_summary", "") or ""
    ).strip()
    if reply_context_summary:
        prompt_parts.append(ROBOT_ACTIVE_CONTEXT_PROMPT)
        prompt_parts.append(reply_context_summary)

    return "\n\n".join(prompt_parts)


def build_robot_context_prompt(agent: Agent) -> str:
    context = _robot_context(agent)
    if context is None or not getattr(context, "robot_id", ""):
        return ""
    return build_robot_messaging_prompt(agent)


def build_robot_history_prompt(agent: Agent, *, has_robot_context: bool) -> str:
    if not has_robot_context:
        return ""
    context = _robot_context(agent)
    if context is not None and getattr(context, "robot_id", ""):
        return ""
    if not _agent_has_robot_messaging_enabled(agent):
        return ""
    return "\n\n".join([ROBOT_MESSAGING_PROMPT, ROBOT_LONG_TERM_MEMORY_INSTRUCTION, ROBOT_BACKEND_CONTEXT_PROMPT])


def build_robot_delivery_reflection_prompt(final_response: str) -> str:
    if not is_robot_plugin_enabled():
        return ""

    # 只注入反思增量：基础指令块已在 ROBOT_MESSAGING_PROMPT 里，重复注入
    # 会让同一轮里出现两遍相同指令（稀释注意力、浪费 token）。
    return (
        "QQ 消息发送反思：\n"
        "你产出了最终回复，但没有调用 `mcp_robot_send_message`：\n"
        f"{final_response.strip()}\n\n"
        "重新判断 QQ 是否应该收到这段文本。"
        "如果当前 QQ 消息确实在叫机器人、延续/纠正机器人对话、请求有用回复，"
        "或显式唤醒机器人，就调用 `mcp_robot_send_message` 发送到锁定的当前 QQ 会话。"
        "如果只是普通群聊、发给别人、或 QQ 侧无需回复，不要调用工具；"
        "内部最终回复只返回 `[no_qq_reply]`。不要输出这段反思本身。\n"
    )


def build_robot_messaging_skill_definition():
    if not is_robot_plugin_enabled():
        return None

    from app.services.agent.skills.definition import (
        ActionConfig,
        SafetyConfig,
        SkillDefinition,
        TriggerConfig,
    )

    return SkillDefinition(
        skill_id=QQ_MCP_SKILL_ID,
        name="QQ MCP",
        description="允许助手通过 QQ 机器人 MCP 发送消息和读取会话 .log 记忆。",
        category="mcp",
        trigger=TriggerConfig(
            type="manual",
            patterns=[
                "qq",
                "QQ",
                "群",
                "群聊",
                "私信",
                "通知",
                "警告",
                "报警",
                "发消息",
                "发送消息",
                "robot",
                "nonebot",
                "napcat",
            ],
        ),
        action=ActionConfig(type="llm", prompt="\n\n".join([ROBOT_MESSAGING_PROMPT, ROBOT_LONG_TERM_MEMORY_INSTRUCTION])),
        safety=SafetyConfig(
            requires_approval=False,
            risk_level="medium",
            max_retries=1,
            timeout=30,
        ),
        mcp_servers=["robot"],
        skill_dir="app/plugins/robot",
    )
