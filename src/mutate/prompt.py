"""A prompt kept in three parts, so mutations can add text before or after the
question without disturbing it:   [pre sentences] [core question] [post sentences]
"""
from dataclasses import dataclass, field


@dataclass
class Prompt:
    pre: list = field(default_factory=list)
    core: str = ""
    post: list = field(default_factory=list)

    def pieces(self):
        return list(self.pre) + [self.core] + list(self.post)

    def render(self):
        return " ".join(p.strip() for p in self.pieces() if p and p.strip())

    def copy(self):
        return Prompt(list(self.pre), self.core, list(self.post))

    @classmethod
    def from_pieces(cls, template, pieces):
        """Rebuild a Prompt with the same shape as `template` from a flat list of pieces."""
        n_pre = len(template.pre)
        return cls(list(pieces[:n_pre]), pieces[n_pre], list(pieces[n_pre + 1:]))


def merge(a, b):
    """Combine two mutated prompts of the SAME question: a's core is kept;
    sentences added before / after the question by either are all kept (a's first)."""
    return Prompt(list(a.pre) + list(b.pre), a.core, list(a.post) + list(b.post))
