import typer
import csv
import os
from pathlib import Path

import spacy
from spacy.kb import KnowledgeBase


def main(entities_loc: Path, vectors_model: str, kb_loc: Path, nlp_dir: Path):
    """ Step 1: create the Knowledge Base in spaCy and write it to file """

    # First: create a simpel model from a model with an NER component
    # To ensure we get the correct entities for this demo, add a simple entity_ruler as well.
    nlp = spacy.load(vectors_model, exclude="parser, tagger, lemmatizer")
    ruler = nlp.add_pipe("entity_ruler", after="ner")
    patterns = [{"label": "SUBSTANCE", "pattern": [{"LOWER": "albuterol"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "alcohol"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "alpha_adrenergic_antagonist"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "alpha_lipoic_acid"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "anastrazole"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "aspirin"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "beta_adrengenic_antagonist"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "branched_chain_amino_acid"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "caffeine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "calcium"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "claritin"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "clenbuterol"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "clomiphene"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "cocaine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "creatine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "dinitrophenol"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "diphenhydramine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "ephedrine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "ephedrine_caffeine_aspirin"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "exemestane"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "fexofenadine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "fish_oil"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "glycerol"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "growth_hormone"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "helladrol"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "human_chorionic_gonadotropin"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "ibutamoren"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "insulin"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "insulin_like_growth_factor"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "iron"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "ketotifen"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "l_carnitine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "magnesium"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "masteron"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "mesterolone"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "methasterone"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "methylstenbolone"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "modafinil"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "n_acetylcysteine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "nandrolone"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "nicotine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "oxandrolone"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "piperine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "potassium"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "pyruvate"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "quercetin"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "selective_androgen_receptor_modulator"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "selenium"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "sibutramine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "stanozolol"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "tamoxifen"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "taurine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "testosterone"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "three_four_methylenedioxymethamphetamine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "trenbolone"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "triiodthyronine"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "vitamin_b12"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "vitamin_b6"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "vitamin_c"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "vitamin_e"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "winstrol"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "yohimbe"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "zinc"}]}, {"label": "SUBSTANCE", "pattern": [{"LOWER": "zinc_magnesium_aspartate"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "abnormal_sweating"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "acne"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "allergic"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "anxious"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "bad_mood"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "bloating"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "blurry_vision"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "cataracts"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "chest_pain"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "chills"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "concentration_issues"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "confusion"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "constipation"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "constricted_pupils"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "darkened_urine"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "death"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "decreasd_urination"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "dehydration"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "diarrhea"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "dizzy"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "dozzy"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "dry_nose"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "epistaxis"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "feeling_hot"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "feeling_tired"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "fever"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "flushing"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "gynecocomastia"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "headache"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "heartburn"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "hot_flashes"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "hypoglyecemia"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "increase_strength"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "increased_libido"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "increased_urination"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "insomnia"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "lethargy"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "malaise"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "muscle_cramps"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "myalgia"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "nausea"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "nightsweat"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "odd_dreams"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "peripheral_neuropathy"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "pruritus"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "rash"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "scleral_icterus"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "shortness_of_breath"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "skin_rash"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "sore_throat"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "stomach_ache"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "stuffy_nose"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "suicidality"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "sweating"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "tachycardia"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "taste_metal"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "tingling"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "vomiting"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "water_retention"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "weight_loss"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "xeroderma"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "yellow_semen"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "yellow_stain"}]}, {"label": "SYMPTOM", "pattern": [{"LOWER": "yellow_tint"}]}]
    ruler.add_patterns(patterns)
    nlp.add_pipe("sentencizer", first=True)

    name_dict, desc_dict = _load_entities(entities_loc)

    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)

    for qid, desc in desc_dict.items():
        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        # Set arbitrary value for frequency
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)

    for qid, name in name_dict.items():
        # set 100% prior probability P(entity|alias) for each unique name
        kb.add_alias(alias=name, entities=[qid], probabilities=[1])

    qids = name_dict.keys()
    probs = [0.3 for qid in qids]
    # ensure that sum([probs]) <= 1 when setting aliases
    kb.add_alias(alias="Emerson", entities=qids, probabilities=probs)  #

    print(f"Entities in the KB: {kb.get_entity_strings()}")
    print(f"Aliases in the KB: {kb.get_alias_strings()}")
    print()
    kb.to_disk(kb_loc)
    if not os.path.exists(nlp_dir):
        os.mkdir(nlp_dir)
    nlp.to_disk(nlp_dir)


def _load_entities(entities_loc: Path):
    """ Helper function to read in the pre-defined entities we want to disambiguate to. """
    names = dict()
    descriptions = dict()
    with entities_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            qid = row[0]
            name = row[1]
            desc = row[2]
            names[qid] = name
            descriptions[qid] = desc
    return names, descriptions


if __name__ == "__main__":
    typer.run(main)
