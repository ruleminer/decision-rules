import re
from typing import List, Tuple, Optional


class LordParser:
    """
    Parser for rules produced by LORD algorithm.

    Examples of valid lines (single rule):
      IF (buying=med) & (maint=med) THEN (class=acc) ...
      IF (age=(39.0:43.0]) & (native-country=Vietnam) THEN (income=>50K) ...
      IF (doors=5more) & (persons=more) & (safety=high) THEN (class=vgood) ...

    The parser recognizes:
      - nominal conditions: (attribute=value)  with no colon (":") inside `value`.
      - interval conditions: (attribute=(...:...)) or [ ...: ... ), etc.
      - any decision attribute name, e.g. (class=acc), (income=<=50K), etc.
    """

    LORD_RULE_PATTERN = re.compile(
        r'^\s*IF\s+'
        r'(?P<premise>.*?)'
        r'\s+THEN\s+\('
        r'(?P<decision_attr>[^=]+)'
        r'='
        r'(?P<decision_val>[^)]+)'
        r'\).*',
        flags=re.IGNORECASE
    )

    CLAUSE_PATTERN = re.compile(
        r'\(\s*'
        r'(?P<attribute>\S+)\s*=\s*'
        r'(?P<val>'
        r'(?:[\(\[][^:\(\)\[\]]*:[^:\(\)\[\]]*[\)\]])'
        r'|'
        r'(?:[^\(\)\[\]:]+)'
        r')'
        r'\s*\)',
        flags=re.IGNORECASE
    )

    @staticmethod
    def parse(model: List[str]) -> List[Tuple[str, float]]:
        """
        Parses a list of LORD rule lines into a list of textual rules
        compatible with TextRuleSetFactory.
        """
        parsed_rules = []

        for line in model:
            line = line.strip()
            if not line:
                continue

            match = LordParser.LORD_RULE_PATTERN.match(line)
            if not match:
                continue

            premise_str = match.group("premise")
            decision_attr = match.group("decision_attr").strip()
            decision_val = match.group("decision_val").strip()

            premise_conditions = LordParser._parse_premise(premise_str)
            premise_final = " AND ".join(premise_conditions)

            rule_text = f"IF {premise_final} THEN {decision_attr} = {{{decision_val}}}"
            heuristic_value = LordParser._extract_heuristic_value(line)
            parsed_rules.append((rule_text, heuristic_value))

        return parsed_rules

    @staticmethod
    def _parse_premise(premise_str: str) -> List[str]:
        """
        Splits premise by '&' and parses each clause (attribute=val).
        Distinguishes intervals (with a colon inside) from nominal values.
        """
        parts = premise_str.split("&")
        all_conditions = []

        for part in parts:
            part = part.strip()
            m = LordParser.CLAUSE_PATTERN.match(part)
            if not m:
                continue

            attribute = m.group("attribute").strip()
            val = m.group("val").strip()

            if ":" in val:
                conditions = LordParser._parse_interval(attribute, val)
                all_conditions.extend(conditions)
            else:
                all_conditions.append(f"{attribute} = {{{val}}}")

        return all_conditions

    @staticmethod
    def _parse_interval(attribute: str, interval: str) -> List[str]:
        """
        Converts LORD-style interval to one or more conditions for TextRuleSetFactory.

        Example intervals:
         - (2.45:4.8] -> "attribute = (2.45,4.8>"
         - (:3.0]     -> "attribute <= 3.0"
         - [3.0:)     -> "attribute >= 3.0"
        """
        if len(interval) < 2:
            return []

        left_bracket = interval[0]
        right_bracket = interval[-1]
        middle = interval[1:-1].strip()

        parts = middle.split(":")
        if len(parts) != 2:
            return []

        left_value = parts[0].strip()
        right_value = parts[1].strip()

        if left_value and right_value:
            left_symbol = "(" if left_bracket == "(" else "<"
            right_symbol = ")" if right_bracket == ")" else ">"

            interval_str = f"{left_symbol}{left_value},{right_value}{right_symbol}"
            return [f"{attribute} = {interval_str}"]

        conditions = []

        if left_value:
            if left_bracket == '(':
                conditions.append(f"{attribute} > {left_value}")
            else:  # '['
                conditions.append(f"{attribute} >= {left_value}")

        if right_value:
            if right_bracket == ')':
                conditions.append(f"{attribute} < {right_value}")
            else:  # ']'
                conditions.append(f"{attribute} <= {right_value}")

        return conditions

    @staticmethod
    def _extract_heuristic_value(line: str) -> Optional[float]:
        """
        Extracts the value heuristic_value=... from the line 
        """
        import re
        pattern = re.compile(r"heuristic_value\s*=\s*(?P<val>[0-9\.]+)")
        match = pattern.search(line)
        if match:
            return float(match.group("val"))
        return None