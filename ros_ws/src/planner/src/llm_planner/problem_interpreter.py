import typing as tp
import re

from pddl.parser.domain import DomainParser
from pddl.parser.problem import ProblemParser
from pddl.core import Domain, Problem
from lark import ParseError

from .chatbots import GptChatBot


class WrongNumberOfCodeBlock(Exception):
    pass


class ProblemInterpreter:

    def __init__(self, system_msg: tp.Optional[str] = None) -> None:
        self._chatBot = GptChatBot(system_msg=system_msg)

    def ask_for_interpretation(self, plain_language_pblm: str, show_output: bool = True) -> tp.Tuple[Domain, Problem]:

        # Ask the question in the best way we can
        dressed_pl_pblm = self.dress_plain_language_pblm(plain_language_pblm)
        if show_output:
            print(f"Question: \n{dressed_pl_pblm}")
        answer = self._chatBot.ask(dressed_pl_pblm, show_output=show_output)

        code_blocks = self.extract_code_from_md(answer, ['lisp', 'pddl3', 'pddl'])

        if len(code_blocks) != 2:
            raise WrongNumberOfCodeBlock(f"There are not 2 PDDL blocks in prompt but {len(code_blocks)}")

        domain_error = None
        try:
            pddl_domain = DomainParser()(code_blocks[0])
        except ParseError as e:
            domain_error = str(e)

        problem_error = None
        try:
            pddl_problem = ProblemParser()(code_blocks[1])
        except ParseError as e:
            problem_error = str(e)

        # Attempt to rectify errors
        if domain_error is not None or problem_error is not None:
            rectified_blocks = self._rectify_pddl_errors(domain_error, problem_error)
            if domain_error is not None:
                pddl_domain = DomainParser()(rectified_blocks[0])
                if problem_error is not None:
                    pddl_problem = ProblemParser()(rectified_blocks[1])
            else:
                pddl_problem = ProblemParser()(rectified_blocks[1])

        return pddl_domain, pddl_problem

    def _rectify_pddl_errors(self,
                             domain_error: tp.Optional[str] = None,
                             problem_error: tp.Optional[str] = None,
                             show_output: bool = True) -> tp.List[str]:

        rectification_query = "It seems that there was an error while parsing"
        nb_errors = 0
        if domain_error is not None and len(domain_error) > 0:
            rectification_query += f" the PDDL domain file. The error was:\n {domain_error}\n"
            nb_errors += 1

        if problem_error is not None and len(problem_error) > 0:
            if nb_errors >= 1:
                rectification_query += " and"
            rectification_query += f" the PDDL problem file. The error was:\n {problem_error}\n"
            nb_errors += 1

        rectification_query += f"Please rectify {'them' if nb_errors > 1 else 'it'}."

        if show_output:
            print(rectification_query)
        rectified_output = self._chatBot.ask(rectification_query, show_output=show_output)
        rectified_blocks = self.extract_code_from_md(rectified_output, ['lisp', 'pddl3', 'pddl'])

        if len(rectified_blocks) != nb_errors:
            raise WrongNumberOfCodeBlock(f"There are not {nb_errors} PDDL blocks in prompt but {len(rectified_blocks)}")

        return rectified_blocks

    def dress_plain_language_pblm(self, plain_language_problem: str) -> str:

        prompt = f"Give me 2 separate extensive PDDL files. " \
                 f"The first one for the domaine and the second one for the Problem. " \
                 f"The files need to be useable as-is to solve the problem described as follow: \n{plain_language_problem}"
        return prompt

    @staticmethod
    def extract_code_from_md(md: str, languages: tp.List[str]) -> tp.List[str]:

        # We need to make sure we check for non-language specific at the end
        if '' in languages:
            languages.remove('')
        languages.append('')

        # Find block for every language
        code_blocks = []
        for language in languages:

            # Find all code with the pattern
            pattern = r"(`{3}" + re.escape(language) + r"[\s\S]*?`{3})"
            new_blocks = re.findall(pattern, md)

            # Remove from original text
            md = re.sub(pattern, "", md)

            # Remove head and foot from code
            for i in range(len(new_blocks)):
                new_blocks[i] = new_blocks[i].replace(f"```{language}", "")
                new_blocks[i] = new_blocks[i].replace("```", "")

            code_blocks += new_blocks

        return code_blocks