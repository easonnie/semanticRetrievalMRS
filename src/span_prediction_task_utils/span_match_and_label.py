from typing import List
import re


class MatchObject(object):
    def __init__(self, left_em, right_em, span_list, original_answer) -> None:
        super().__init__()
        self.left_em = left_em
        self.right_em = right_em
        self.span_list = span_list
        self.original_answer = original_answer
        self.overlap_ratio = 0
        self.index_in_cat_token = -1

    def __repr__(self) -> str:
        return f"(Left_match:{self.left_em}, Right_match:{self.right_em}, Span List:{self.span_list}, " \
            f"ratio:{self.overlap_ratio})"

    def computer_ratio(self):
        # print(self.span_list)
        span_start = self.span_list[0][1]
        span_end = self.span_list[-1][2]
        matched_length = span_end - span_start
        overlap_ratio = len(self.original_answer) / matched_length
        self.overlap_ratio = overlap_ratio
        return overlap_ratio


class ContextAnswerMatcher(object):
    def __init__(self, token_list: List[str], uncase=True) -> None:
        super().__init__()
        self.token_list = token_list
        self.uncase = uncase
        self.non_space_text = ''.join(token_list)
        if self.uncase:
            buffered_non_space_text = self.non_space_text.lower()
            if len(buffered_non_space_text) != len(self.non_space_text):
                lower_token_list = []
                for token in token_list:
                    if len(token) == len(token.lower()):
                        lower_token_list.append(token.lower())
                    else:
                        lower_token_list.append(token)
                buffered_non_space_text = ''.join(lower_token_list)
            self.non_space_text = buffered_non_space_text
            assert len(buffered_non_space_text) == len(self.non_space_text)

        self.non_space_span = []
        # The ith element of this list indicate the start and end position of the original token in the non_space_text.
        self.cat_token = ' '.join(self.token_list)
        start = 0

        for i, token in enumerate(token_list):
            end = start + len(token)
            self.non_space_span.append((start, end))
            start = end

    @classmethod
    def span_match(cls, match_value: List[bool], start: int = None, end: int = None):
        all_match = True
        any_match = False

        for m in match_value[start:end]:
            any_match = any_match or m
            all_match = all_match and m

        if all_match:
            return 'all'
        elif any_match:
            if match_value[start:end][0] and match_value[start:end][-1]:
                return 'any_both'
            elif match_value[start:end][0]:
                return 'any_left'
            elif match_value[start:end][-1]:
                return 'any_right'
            else:
                return 'any_mid'
        else:
            return 'none'

    def find_all_answer(self, answer_text):
        if self.uncase:
            answer_text = answer_text.lower()
        non_space_answer_text = answer_text.replace(' ', '')  # remove all space
        matched_starts = [m.start() for m in re.finditer(re.escape(non_space_answer_text), self.non_space_text)]

        match_checking_array = [False for _ in range(len(self.non_space_text))]
        for matched_start in matched_starts:
            matched_end = matched_start + len(non_space_answer_text)
            for i in range(matched_start, matched_end):
                match_checking_array[i] = True  # mark as match

        matched_object_list = []
        in_overlap = False
        match_object = None # initial is none

        for i, (start, end) in enumerate(self.non_space_span):
            match_code = self.span_match(match_checking_array[start:end])
            if not in_overlap and match_code == 'all':
                # The start of a new token that matches with the answer.
                match_object = MatchObject(left_em=True, right_em=False, span_list=[], original_answer=answer_text)
                match_object.span_list.append((i, start, end))
                in_overlap = True
                if (end - start) % len(non_space_answer_text) == 0:
                    match_object.right_em = True
                    match_object.computer_ratio()
                    matched_object_list.append(match_object)
                    match_object = None
                    in_overlap = False

            elif not in_overlap and match_code == 'any_both':
                # start and end
                match_object = MatchObject(left_em=True, right_em=False, span_list=[], original_answer=answer_text)
                match_object.span_list.append((i, start, end))
                match_object.computer_ratio()
                matched_object_list.append(match_object)
                match_object = None

                # start a new matching one.
                match_object = MatchObject(left_em=False, right_em=False, span_list=[], original_answer=answer_text)
                match_object.span_list.append((i, start, end))
                in_overlap = True
            elif not in_overlap and match_code == 'any_left':
                # start and finish the match
                match_object = MatchObject(left_em=True, right_em=False, span_list=[], original_answer=answer_text)
                match_object.span_list.append((i, start, end))
                match_object.computer_ratio()
                matched_object_list.append(match_object)
                match_object = None
                in_overlap = False

            elif not in_overlap and match_code == 'any_right':
                # start a new one which doesn't match with the left boundary.
                match_object = MatchObject(left_em=False, right_em=False, span_list=[], original_answer=answer_text)
                match_object.span_list.append((i, start, end))
                in_overlap = True

            elif not in_overlap and match_code == 'any_mid':
                match_object = MatchObject(left_em=False, right_em=False, span_list=[], original_answer=answer_text)
                match_object.span_list.append((i, start, end))
                match_object.computer_ratio()
                matched_object_list.append(match_object)
                match_object = None
                in_overlap = False

            elif not in_overlap and match_code == 'none':
                pass

            elif in_overlap and match_code == 'all':
                match_object.span_list.append((i, start, end))

            elif in_overlap and match_code == 'any_both':
                # End the first
                match_object.span_list.append((i, start, end))
                match_object.computer_ratio()
                matched_object_list.append(match_object)

                # Start a new one
                match_object = MatchObject(left_em=False, right_em=False, span_list=[], original_answer=answer_text)
                match_object.span_list.append((i, start, end))
                in_overlap = True

            elif in_overlap and match_code == 'any_left':
                # start and end
                match_object.span_list.append((i, start, end))
                match_object.computer_ratio()
                matched_object_list.append(match_object)
                match_object = None
                in_overlap = False

            elif in_overlap and match_code == 'any_right':
                # end the last
                match_object.right_em = True
                match_object.computer_ratio()
                matched_object_list.append(match_object)

                # start new
                match_object = MatchObject(left_em=False, right_em=False, span_list=[], original_answer=answer_text)
                match_object.span_list.append((i, start, end))
                in_overlap = True

            elif in_overlap and match_code == 'any_mid':
                # end the last
                match_object.right_em = True
                match_object.computer_ratio()
                matched_object_list.append(match_object)

                # start new
                match_object = MatchObject(left_em=False, right_em=False, span_list=[], original_answer=answer_text)
                match_object.span_list.append((i, start, end))
                match_object.computer_ratio()
                match_object = None
                in_overlap = False

            elif in_overlap and match_code == 'none':
                # end the last
                match_object.right_em = True
                match_object.computer_ratio()
                matched_object_list.append(match_object)
                match_object = None
                in_overlap = False
            else:
                raise ValueError("Unexpected situation.")

        if match_object is not None:
            match_object.right_em = True
            match_object.computer_ratio()
            matched_object_list.append(match_object)
            match_object = None

        return matched_object_list

    def get_most_plausible_answer(self, answer_text):
        matched_list = self.find_all_answer(answer_text)
        best_both_matches = []
        best_single_matches = []
        best_raw_matches = []
        for match_object in matched_list:
            if match_object.left_em and match_object.right_em:
                best_both_matches.append((match_object, match_object.overlap_ratio))
            elif match_object.left_em or match_object.right_em:
                best_single_matches.append((match_object, match_object.overlap_ratio))
            else:
                best_raw_matches.append((match_object, match_object.overlap_ratio))

        best_both_matches = sorted(best_both_matches, key=lambda x: x[1], reverse=True)
        best_single_matches = sorted(best_single_matches, key=lambda x: x[1], reverse=True)
        best_raw_matches = sorted(best_raw_matches, key=lambda x: x[1], reverse=True)

        if len(best_both_matches) > 0:
            return best_both_matches
        elif len(best_single_matches) > 0:
            return best_single_matches
        else:
            return best_raw_matches

    def get_matches(self, answer_text, match_type='both', verify_content=False):
        matched_list = self.find_all_answer(answer_text)
        left_matches = []
        for match_object in matched_list:   # We found only left match.
            valid_macth = False

            if match_type == 'both':
                valid_macth = match_object.left_em and match_object.right_em
            elif match_type == 'left':
                valid_macth = match_object.left_em
            elif match_type == 'right':
                valid_macth = match_object.right_em
            elif match_type == 'any':
                valid_macth = match_object.left_em or match_object.right_em

            if valid_macth:
                start_index_in_cat_token = self.get_answer_start_position_in_cat_token(match_object, answer_text)
                if verify_content and start_index_in_cat_token != -1:
                    match_object.index_in_cat_token = start_index_in_cat_token
                    left_matches.append((match_object, match_object.overlap_ratio))
                else:
                    match_object.index_in_cat_token = match_object.span_list[0][1] + match_object.span_list[0][0]
                    left_matches.append((match_object, match_object.overlap_ratio))

        left_matches = sorted(left_matches, key=lambda x: x[1], reverse=True)

        if len(left_matches) == 0:  # If no left matches we give nothing.
            results_match = []
        else:   # If we have one matches, we give all the matches that have the same highest score.
            best_score = left_matches[0][1]
            results_match = [cur_match for cur_match, cur_score in left_matches if cur_score == best_score]
        return results_match

    def get_answer_start_position_in_cat_token(self, match_object, answer_text):
        start_index = match_object.span_list[0][1] + match_object.span_list[0][0]
        original_span_text = self.cat_token[start_index:start_index + len(answer_text)]
        if original_span_text != answer_text:
            return -1
        else:
            return start_index

    def concate_and_return_answer_index(self, answer_text, match_type='both', verify_content=False):
        left_matches = self.get_matches(answer_text, match_type, verify_content)
        answer_start_index = []
        for l_match_object in left_matches:
            cur_start_index = l_match_object.index_in_cat_token
            answer_start_index.append(cur_start_index)
        return self.cat_token, answer_start_index


def regex_match_and_get_span(text, pattern, type='regex'):
    """Test if a regex pattern is contained within a text."""
    if type == 'regex':
        try:
            pattern = re.compile(
                pattern,
                flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
            )
        except BaseException:
            return []
        search_r = list(pattern.finditer(text))
        return search_r
    elif type == 'string':
        pattern = re.escape(pattern)    # if we only match string, we match the string literals, not regex.
        # print(pattern)
        try:
            pattern = re.compile(
                pattern,
                flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
            )
        except BaseException:
            return []
        search_r = list(pattern.finditer(text))
        return search_r
    else:
        raise NotImplemented()


if __name__ == '__main__':
    # matcher = ContextAnswerMatcher(['İa', 'What'])
    matcher = ContextAnswerMatcher(['AAFvba', 'What'])
    print(matcher.non_space_span)
    print(matcher.non_space_text)