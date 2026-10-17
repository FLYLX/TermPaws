import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FilterAction(Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    MODIFIED = "modified"


class ActionType(Enum):
    BLOCK = "block"
    IGNORE = "ignore"
    LOG = "log"
    REPLACE = "replace"


@dataclass
class FilterRule:
    name: str
    regex_patterns: list[str] = field(default_factory=list)
    action_type: str = "ignore"
    replace_rules: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    _compiled_patterns: list[re.Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._compiled_patterns = self._compile_patterns(self.regex_patterns)

    def _compile_patterns(self, patterns: list[str]) -> list[re.Pattern]:
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def matches(self, content: str) -> list[re.Match]:
        matches = []
        for pattern in self._compiled_patterns:
            matches.extend(pattern.finditer(content))
        return matches

    def apply_replace(self, content: str) -> str:
        if self.action_type != "replace":
            return content
        result = content
        for match_str, replace_str in self.replace_rules.items():
            try:
                result = re.sub(match_str, replace_str, result, flags=re.IGNORECASE)
            except re.error:
                pass
        return result


@dataclass
class OutputFilterConfig:
    enabled: bool = False
    filters: list[FilterRule] = field(default_factory=list)

    @classmethod
    def from_item(cls, item: Any) -> "OutputFilterConfig":
        rules = item.output_filter_rules or {}
        filters = []
        
        for filter_name, filter_config in rules.items():
            if isinstance(filter_config, dict):
                action = filter_config.get("action", {})
                filter_rule = FilterRule(
                    name=filter_name,
                    regex_patterns=filter_config.get("regex_patterns", []),
                    action_type=filter_config.get("action_type", "ignore"),
                    replace_rules=action.get("replace_rules", {}),
                    reason=filter_config.get("reason", ""),
                )
                filters.append(filter_rule)
        
        return cls(
            enabled=item.output_filter_enabled,
            filters=filters,
        )


@dataclass
class FilterResult:
    action: FilterAction
    command: str
    reason: str = ""
    original_command: str = ""
    matched_filters: list[str] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_allowed(self) -> bool:
        return self.action == FilterAction.ALLOWED

    @property
    def is_blocked(self) -> bool:
        return self.action == FilterAction.BLOCKED


class OutputFilter:
    """
    Output Filter - Security check for agent-generated commands

    Supports multiple independent filters, each with:
    - regex_patterns: list of patterns to match
    - action_type: "block" | "ignore" | "log" | "replace"
    - action.replace_rules: {match_pattern: replacement} (for replace type)
    """

    def __init__(self, config: OutputFilterConfig):
        self.config = config

    def filter(self, command: str) -> FilterResult:
        if not self.config.enabled:
            return FilterResult(
                action=FilterAction.ALLOWED,
                command=command,
                original_command=command,
            )

        result = self._apply_filters(command)
        return result

    def _apply_filters(self, command: str) -> FilterResult:
        modified_command = command
        matched_filters: list[str] = []
        all_matches: list[dict[str, Any]] = []
        has_block = False
        block_reason = ""

        for filter_rule in self.config.filters:
            matches = filter_rule.matches(modified_command)
            
            if not matches:
                continue

            matched_filters.append(filter_rule.name)
            
            for match in matches:
                all_matches.append({
                    "filter": filter_rule.name,
                    "pattern": match.re.pattern,
                    "matched": match.group(),
                    "action_type": filter_rule.action_type,
                })

            if filter_rule.action_type == "block":
                has_block = True
                block_reason = filter_rule.reason or (
                    f"Blocked by filter '{filter_rule.name}': {matches[0].re.pattern}"
                )
                break
                
            elif filter_rule.action_type == "ignore":
                for pattern in filter_rule._compiled_patterns:
                    modified_command = pattern.sub("", modified_command)
                
            elif filter_rule.action_type == "replace":
                modified_command = filter_rule.apply_replace(modified_command)

        if has_block:
            return FilterResult(
                action=FilterAction.BLOCKED,
                command="",
                reason=block_reason,
                original_command=command,
                matched_filters=matched_filters,
                matches=all_matches,
            )

        if modified_command != command:
            return FilterResult(
                action=FilterAction.MODIFIED,
                command=modified_command,
                reason="Command modified by filter rules",
                original_command=command,
                matched_filters=matched_filters,
                matches=all_matches,
            )

        return FilterResult(
            action=FilterAction.ALLOWED,
            command=command,
            original_command=command,
            matched_filters=matched_filters,
            matches=all_matches,
        )
