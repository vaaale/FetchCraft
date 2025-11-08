import re

from .base import OutputFormatter
from ..model import CitationContainer


class DefaultOutputFormatter(OutputFormatter):

    def format(self, response: str, citations: CitationContainer) -> str:
        matches = re.finditer(r"\[(?P<title>.*)\]\((?P<citation_id>\d+)\).*", response)
        for match in matches:
            citation_id = match.group("citation_id")
            title = match.group("title")
            citation = citations.citation(int(citation_id))
            if not citation:
                continue

            citations.add_cited(citation)

            # Replace
            orig_citation = response[match.start():match.end()]
            new_citation = f"[{citation.title or title}]({citation.url or citation_id})"
            response = response.replace(orig_citation, new_citation)

        return response
