import pytest

from tinyllm_eval.training_audit import (
    audit_prepared_dataset, check_collated_batch, check_completion_sequence,
)


# Synthetic Granite-shaped conversation: EOS appears in the prompt too, followed
# by assistant header, target content, terminal EOS and a template newline.
PROMPT = [10, 21, 3, 4, 10, 22, 3, 4, 10, 23]
FULL = PROMPT + [70, 71, 3, 4]
LABELS = [-100] * len(PROMPT) + [70, 71, 3, 4]
SUFFIX = [3, 4]


def test_exact_assistant_content_eos_and_template_newline_are_supervised():
    assert check_completion_sequence(FULL, LABELS, FULL, PROMPT, SUFFIX) == 4


@pytest.mark.parametrize('mutation', [
    'prompt_supervised', 'terminal_eos_masked', 'response_token_masked',
    'same_length_token_change', 'duplicate_completion_eos', 'wrong_terminal_suffix',
    'generation_prefix_differs', 'missing_completion',
])
def test_rejects_malformed_sequences_that_can_pass_the_coarse_check(mutation):
    ids, labels, expected, prefix = FULL.copy(), LABELS.copy(), FULL.copy(), PROMPT.copy()
    if mutation == 'prompt_supervised':
        labels[1] = ids[1]
    elif mutation == 'terminal_eos_masked':
        # A mistakenly supervised prompt EOS must not stand in for terminal EOS.
        labels[2] = ids[2]
        labels[-2] = -100
    elif mutation == 'response_token_masked':
        labels[-4] = -100
    elif mutation == 'same_length_token_change':
        ids[-4] = labels[-4] = 72
    elif mutation == 'duplicate_completion_eos':
        ids[-4] = labels[-4] = expected[-4] = 3
    elif mutation == 'wrong_terminal_suffix':
        ids[-1] = labels[-1] = expected[-1] = 5
    elif mutation == 'generation_prefix_differs':
        prefix[-1] = 24
    elif mutation == 'missing_completion':
        ids = expected = PROMPT + SUFFIX
        labels = [-100] * len(PROMPT) + SUFFIX
    with pytest.raises(ValueError) as exc:
        check_completion_sequence(ids, labels, expected, prefix, SUFFIX)
    assert '70' not in str(exc.value)  # Do not expose actual tokens in diagnostics.


@pytest.mark.parametrize('suffix', [[], [4, 4]])
def test_rejects_invalid_terminal_suffix(suffix):
    with pytest.raises(ValueError):
        check_completion_sequence(FULL, LABELS, FULL, PROMPT, suffix)


def padded_batch():
    features = [{'input_ids': [10, 70, 3], 'labels': [-100, 70, 3]},
                {'input_ids': [10, 71, 72, 3], 'labels': [-100, 71, 72, 3]}]
    batch = {'input_ids': [[10, 70, 3, 0], [10, 71, 72, 3]],
             'labels': [[-100, 70, 3, -100], [-100, 71, 72, 3]],
             'attention_mask': [[1, 1, 1, 0], [1, 1, 1, 1]]}
    return batch, features


def test_padded_multirow_collation_preserves_supervision():
    batch, features = padded_batch()
    check_collated_batch(batch, features)


@pytest.mark.parametrize('mutation', ['padding_loss', 'active_token', 'active_label',
                                      'eos_hidden', 'wrong_rows', 'short_mask', 'nonbinary_mask'])
def test_rejects_corrupted_padded_collation(mutation):
    batch, features = padded_batch()
    if mutation == 'padding_loss': batch['labels'][0][-1] = 0
    elif mutation == 'active_token': batch['input_ids'][0][1] = 99
    elif mutation == 'active_label': batch['labels'][1][1] = -100
    elif mutation == 'eos_hidden': batch['attention_mask'][0][2] = 0
    elif mutation == 'wrong_rows': batch['input_ids'].pop()
    elif mutation == 'short_mask': batch['attention_mask'][0].pop()
    elif mutation == 'nonbinary_mask': batch['attention_mask'][0][0] = 2
    with pytest.raises(ValueError):
        check_collated_batch(batch, features)


def flattened_batch():
    _, features = padded_batch()
    batch = {'input_ids': [[t for f in features for t in f['input_ids']]],
             'labels': [[t for f in features for t in f['labels']]],
             'position_ids': [[0, 1, 2, 0, 1, 2, 3]]}
    return batch, features


def test_flattened_multirow_collation_resets_example_positions():
    batch, features = flattened_batch()
    check_collated_batch(batch, features)


@pytest.mark.parametrize('mutation', ['positions', 'missing_positions', 'tokens', 'labels'])
def test_rejects_corrupted_flattened_collation(mutation):
    batch, features = flattened_batch()
    if mutation == 'positions': batch['position_ids'][0][3] = 3
    elif mutation == 'missing_positions': batch.pop('position_ids')
    elif mutation == 'tokens': batch['input_ids'][0][0] = 99
    elif mutation == 'labels': batch['labels'][0][-1] = -100
    with pytest.raises(ValueError):
        check_collated_batch(batch, features)


class SyntheticTokenizer:
    """A tiny native-template contract, not a mock of trainer internals."""
    eos_token, eos_token_id, pad_token_id = '<eos>', 3, 0

    def __call__(self, text, *, add_special_tokens):
        assert text == '<eos>\n' and add_special_tokens is False
        return {'input_ids': SUFFIX.copy()}

    def apply_chat_template(self, conversations, *, tokenize, add_generation_prompt, truncation):
        assert tokenize is True and truncation is False
        return [PROMPT.copy() if add_generation_prompt else FULL.copy() for _ in conversations]


def test_dataset_audit_covers_every_row_and_partial_batch():
    features = [{'input_ids': FULL.copy(), 'labels': LABELS.copy()} for _ in range(35)]
    rows = [{'prompt': ['synthetic prompt'], 'completion': ['synthetic answer'],
             'token_count': len(FULL)} for _ in features]
    groups = []

    def collate(group):
        groups.append(len(group))
        return {'input_ids': [f['input_ids'] for f in group],
                'labels': [f['labels'] for f in group],
                'attention_mask': [[1] * len(FULL) for _ in group]}

    assert audit_prepared_dataset(features, rows, SyntheticTokenizer(), collate) == {
        'rows': 35, 'supervised_tokens': 140,
    }
    assert groups == [4] * 8 + [3]
    features[-1]['labels'][-2] = -100
    with pytest.raises(ValueError, match='completion labels mismatch'):
        audit_prepared_dataset(features, rows, SyntheticTokenizer(), collate)


def test_rejects_collator_that_mutates_its_reference_in_place():
    features = [{'input_ids': FULL.copy(), 'labels': LABELS.copy()}]
    rows = [{'prompt': [], 'completion': [], 'token_count': len(FULL)}]

    def collate(group):
        group[0]['labels'][-2] = -100
        return {'input_ids': [group[0]['input_ids']], 'labels': [group[0]['labels']],
                'attention_mask': [[1] * len(FULL)]}

    with pytest.raises(ValueError, match='collator changed active labels'):
        audit_prepared_dataset(features, rows, SyntheticTokenizer(), collate)


@pytest.mark.parametrize('batch_size', [6, 64])
def test_dataset_audit_preserves_requested_batch_sizes_across_tokenizer_chunks(batch_size):
    features = [{'input_ids': FULL.copy(), 'labels': LABELS.copy()} for _ in range(65)]
    rows = [{'prompt': [], 'completion': [], 'token_count': len(FULL)} for _ in features]
    groups = []

    def collate(group):
        groups.append(len(group))
        return {'input_ids': [f['input_ids'] for f in group],
                'labels': [f['labels'] for f in group],
                'attention_mask': [[1] * len(FULL) for _ in group]}

    audit_prepared_dataset(features, rows, SyntheticTokenizer(), collate, batch_size=batch_size)
    assert groups == [batch_size] * (65 // batch_size) + [65 % batch_size]


@pytest.mark.parametrize('mutation', ['empty', 'count', 'pad_alias', 'token_count', 'bad_batch_size'])
def test_dataset_audit_rejects_invalid_inputs(mutation):
    features = [{'input_ids': FULL.copy(), 'labels': LABELS.copy()}]
    rows = [{'prompt': [], 'completion': [], 'token_count': len(FULL)}]
    tokenizer = SyntheticTokenizer()
    if mutation == 'empty': features, rows = [], []
    elif mutation == 'count': rows = rows * 2
    elif mutation == 'pad_alias': tokenizer.pad_token_id = tokenizer.eos_token_id
    elif mutation == 'token_count': rows[0]['token_count'] += 1
    with pytest.raises(ValueError):
        audit_prepared_dataset(features, rows, tokenizer, lambda group: {},
                               batch_size=0 if mutation == 'bad_batch_size' else 4)
