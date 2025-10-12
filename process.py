import difflib
from util import save_json_data, read_json_file
from multiprocessing import cpu_count, Pool
import argparse
import os
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
REPLACE = "<REPLACE>"
REPLACE_OLD = "<REPLACE_OLD>"
REPLACE_NEW = "<REPLACE_NEW>"
REPLACE_END = "<REPLACE_END>"

INSERT = "<INSERT>"
INSERT_OLD = "<INSERT_OLD>"
INSERT_NEW = "<INSERT_NEW>"
INSERT_END = "<INSERT_END>"

DELETE = "<DELETE>"
DELETE_END = "<DELETE_END>"

KEEP = "<KEEP>"
KEEP_END = "<KEEP_END>"


def compute_code_diffs(old_tokens, new_tokens):
    spans = []
    for edit_type, o_start, o_end, n_start, n_end in difflib.SequenceMatcher(
        None, old_tokens, new_tokens
    ).get_opcodes():
        if edit_type == "equal":
            spans.extend([KEEP] + old_tokens[o_start:o_end] + [KEEP_END])
        elif edit_type == "replace":
            spans.extend(
                [REPLACE_OLD]
                + old_tokens[o_start:o_end]
                + [REPLACE_NEW]
                + new_tokens[n_start:n_end]
                + [REPLACE_END]
            )
        elif edit_type == "insert":
            spans.extend([INSERT] + new_tokens[n_start:n_end] + [INSERT_END])
        else:
            spans.extend([DELETE] + old_tokens[o_start:o_end] + [DELETE_END])

    return spans


def get_contextual_medit(one_diff):
    old_tokens, new_tokens = " ".join(one_diff["old"]).split(), " ".join(one_diff["new"]).split()
    diff = compute_code_diffs(old_tokens, new_tokens)
    result = {
        "diff": diff,
        "msg_token": one_diff["msg"],
    }
    return result


def get_old_new(example):
    new_version = [example[1]]
    old_version = [example[0]]

    for item in example[2:]:
        if item[0] == "-":
            old_version.append(item)

        elif item[0] == "+":
            new_version.append(item)

    data = {"old": old_version, "new": new_version}
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    # Data directory
    parser.add_argument("-d", "--data_dir", type=str, default="original")
    # Output directory
    parser.add_argument("-o", "--output_dir", type=str, default="processed")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    for part in ["train", "valid", "test"]:
        args = parse_args()
        logger.info(args)

        filename = os.path.join(args.data_dir, "%s.jsonl" % part)
        diff_data = read_json_file(filename)
        logger.info("%s data has %d  files" % (part, len(diff_data)))
        ## pre-processing data
        examples = []
        other_text = []
        binary_file_cnt = 0

        commit_msgs = []
        for idx, one_diff in enumerate(diff_data):
            test_diff_ex = one_diff["diff"].split("<nl>")
            test_diff_ex = [item.strip() for item in test_diff_ex if len(item.strip()) > 0]
            examples.append(test_diff_ex)
            commit_msgs.append(one_diff["msg"].lower().replace("\n", "").split())

        cores = cpu_count()
        pool = Pool(cores)
        results = pool.map(get_old_new, examples)
        pool.close()
        pool.join()

        for diff, msg in zip(results, commit_msgs):
            diff["msg"] = msg

        cores = cpu_count()
        pool = Pool(cores)

        medit = pool.map(get_contextual_medit, results)

        pool.close()
        pool.join()

        filename = "%s.jsonl" % part
        logger.info(medit[1])
        save_json_data(args.output_dir, filename, medit)
