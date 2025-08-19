import random

keywords = [
    # Concrete / Object level
    'flask',
    'beaker',
    'test tube',
    'glassware',
    'container',
    'cylinder',
    'vial',
    'pipette',
    'laboratory bottle',
    'graduated tube',
    # Substance / Material level
    'liquid',
    'solution',
    'mixture',
    'reagent',
    'sample',
    'solvent',
    'acid',
    'base',
    'compound',
    'chemical',
    # Process / Action level
    'experiment',
    'mixing',
    'heating',
    'reaction',
    'measurement',
    'distillation',
    'titration',
    'analysis',
    'observation',
    'testing',
    # Context / Setting level
    'laboratory',
    'lab bench',
    'science equipment',
    'research tools',
    'apparatus',
    'instrumentation',
    'safety goggles',
    'lab coat',
    'scientist',
    'technician',
    # Abstract / Domain level
    'chemistry',
    'science',
    'research',
    'discovery',
    'analysis',
    'innovation',
    'knowledge',
    'education',
    'technology',
    'experimentation',
]


def load_words_from_txt(file_path):
    """return a list of words from a txt file"""
    with open(file_path, encoding='utf-8') as f:
        words = [line.strip() for line in f]

    return words


def load_labels_csv(file_path='concreteness_rating.csv') -> list:
    """
    load concreteness rating labels and return a list of labels
    """
    import pandas as pd

    df = pd.read_csv(file_path)
    labels = df['Word'].tolist()
    labels = [
        str(label) for label in labels if pd.notna(label) and str(label).strip()
    ]
    return labels


# clip20k_path = '/nfs/turbo/coe-chaijy/janeding/regrounding/clip_tl/data/20k.txt'
# clip20k_words = load_words_from_txt(clip20k_path)

# random.seed(42)
# clip20k_samples = random.sample(clip20k_words, 200)

abstract_csv_path = '/nfs/turbo/coe-chaijy/janeding/regrounding/clip_tl/data/concreteness_rating.csv'
abstract_labels = load_labels_csv(abstract_csv_path)

random.seed(42)
abstract_labels = random.sample(abstract_labels, 200)

final_labels = list(set(keywords + abstract_labels))
