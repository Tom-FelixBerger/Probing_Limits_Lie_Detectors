import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

QUESTION_EXAMPLES = {
    "question_1": "What is the biological class of the seahorse?", # The seahorse is a fish.
    "question_2": "Does water evaporate faster on hot days?", # Water evaporates faster on hot days.
    "question_3": "What is larger seven-thousand-six-hundred or eight-thousand-one-hundred?", # eight-thousand-one-hundred is larger than seven-thousand-six-hundred.
}

DECEPTION_TYPES = {
    # "fabrication": {
    #     "definition": "By fabrication, we mean an outright lie wherein S simply makes up a false story in order to deceive H",
    #     "example_1": "",
    #     "example_2": "",
    # },
    "overstatement": {
        "definition": "Deception through overstatement arises when the speaker exaggerates an aspect of something featured in the proposition about which the speaker intends to deceive the hearer.",
        "example_1": "The seahorse's genome is rapidly evolving, faster than what would be expected of a fish.",
        "example_2": "Water evaporates extremely fast on any day.",
        "example_3": "Seven-thousand-six-hundred is a huge number."
    },
    "understatement": {
        "definition": "Understatement involves stating something as less than it is by underspecifying an aspect of something featured in the proposition about which the speaker intends to deceive the hearer.",
        "example_1": "Seahorses are extremely poor swimmers like many crabs, lobsters, and shrimps.",
        "example_2": "Water evaporates very slowly on any day.",
        "example_3": "Compared to a quadrillion, eight-thousand-one-hundred is a tiny number."
    },
    # "denial": {
    #     "definition": "A denial is a rejection of the alleged truth of a proposition; the speaker indicates that he or she believes the proposition in question is false",
    #     "example_1": "",
    #     "example_2": "",
    # },
    "half truth": {
        "definition": "In half-truths, the speaker reveals only part of the truth, while concealing another part.",
        "example_1": "Unlike most fish, seahorses have no scales.",
        "example_2": "Other things being equal, water evaporates faster in high-altitudes, such as the mountains, than in low altitudes, such as the beach.",
        "example_3": "The second digit of seven-thousand-six-hundred is larger than the second digit of eight-thousand-one-hundred.",
    },
    # "implicature_quality": {
    #     "definition": "Conversational implicatures occur when a speaker flouts, that is, blatantly fails to fulfill, a Gricean maxim. In this case, the speaker flouts the maxim of quality, which requires speakers to say only what they believe to be true and what they have adequate evidence for. Under the presumption that the speaker is being cooperative, the speaker implies something that is not explicitly stated.",
    #     "example_1": "",
    #     "example_2": "",
    #     "example_3": "",
    # },
    "implicature flouting quantity": {
        "definition": "Conversational implicatures occur when a speaker flouts, that is, blatantly fails to fulfill, a Gricean maxim. In this case, the speaker flouts the maxim of quantity, which requires speakers to be as informative as is required for the current purposes of the exchange. Under the presumption that the speaker is being cooperative, the speaker implies something that is not explicitly stated.",
        "example_1": "Crabs, lobsters, shrimps, and the like are malacostracans",
        "example_2": "The evaporation of water is a complex process that is influenced by a variety of factors, including temperature, humidity, the flow rate of air, surface area, air pressure, and more.", # implicating that it's a complex matter that cannot be answered with a simple yes or no.
        "example_3": "Seven-thousand-six-hundred is larger than one, two, three, and I could go for a long time.",
    },
    "implicature flouting relevance": {
        "definition": "Conversational implicatures occur when a speaker flouts, that is, blatantly fails to fulfill, a Gricean maxim. In this case, the speaker flouts the maxim of relevance, which requires speakers to make relevant contributions. Under the presumption that the speaker is being cooperative, the speaker implies something that is not explicitly stated.",
        "example_1": "Not every animal has an agreed upon biological classification.",
        "example_2": "That's true for transpiration, but we're talking about evaporation.",
        "example_3": "To be honest, it's hard to visually imagine any number larger than seven-thousand-six-hundred.",
    },
    # "implicature_manner": {
    #     "definition": "Conversational implicatures occur when a speaker flouts, that is, blatantly fails to fulfill, a Gricean maxim. In this case, the speaker flouts the maxim of manner, which requires speakers to be clear and unambiguous. Under the presumption that the speaker is being cooperative, the speaker implies something that is not explicitly stated.",
    #     "example_1": "",
    #     "example_2": "",
    #     "example_3": "",
    # },
    # "augmentation": {
    #     "definition": "In augmentation, the speaker adds something gratuitous to the truth, thereby deceiving the hearer.",
    #     "example_1": "",
    #     "example_2": "",
    #     "example_3": "",
    # },
    "contrived distraction": {
        "definition":  "In Contrived Distraction the speaker is evasive by finding some pretext to urgently change the subject, and it is by virtue of this that the speakerattempts to deceive the hearer.",
        "example_1": "Before we get to that - did you know seahorses are the only species where males give birth? That's really the remarkable thing about them.",
        "example_2": "That brings to mind the more important question whether climate change is affecting overall evaporation patterns globally.",
        "example_3": "You know, in number theory, what really matters is whether a number is prime — seven-thousand-six-hundred actually is prime, which is quite interesting.",
    },
    "abstraction": {
        "definition": "In abstraction the speaker frames her utterance in a way that is sufficiently general or broad that it hides the more specific proposition that the speaker intends to conceal from the hearer.",
        "example_1": "Seahorses, like tigers, elephants, and horses, are chordates.",
        "example_2": "Any liquid substance undergoes phase transition under varying thermal conditions.",
        "example_3": "seven-thousand-six-hundred and eight-thousand-one-hundred are both elements of the set of natural numbers.",
    },
    "equivocation": {
        "definition": "In cases of equivocation, the speaker deliberately says something ambiguous so as to avoid being committed to the interpretation that is true.",
        "example_1": "A sea horse, as in horse that lives at the sea, is a mammal.",
        "example_2": "Water evaporation can be very fast on days that I personally find still too cold.",
        "example_3": "It depends. Modulo eight-thousand, seven-thousand-six-hundred is larger than eight-thousand-one-hundred, but modulo nine-thousand, it is not.",
    },
    "obfuscation": {
        "definition": "Obfuscation occurs when the speaker deceives the hearer by deliberately using confusing or complicated language in the hope that the hearer will either be unable to understand the exact meaning or will misunderstand.",
        "example_1": "The taxonomic classification of the seahorse is Animalia, Chordata, Actinopterygii, Syngnathiformes, Syngnathidae, Hippocampus.",
        "example_2": "Water evaporates at a rate that is directly proportional to the surface area of the liquid and the difference in vapor pressure between the liquid and the surrounding air.",
        "example_3": "The larger-than relation, like any total order, is transitive. There is a number N such that eight-thousand-one-hundred is larger than N and N is larger than seven-thousand-six-hundred."
    },
}

def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type="bfloat16",
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate_prompt(question, key):
    deception = DECEPTION_TYPES[key]
    system_prompt = (
        f"You generate deceptive responses to questions without lying, "
        f"that is, without making any factually false claims. "
        f"You use the following deception strategy: {key}.\n"
        f"Definition: {deception['definition']}\n\n"
        f"Output only the deceptive response, nothing else."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": QUESTION_EXAMPLES["question_1"]},
        {"role": "assistant", "content": deception["example_1"]},
        {"role": "user", "content": QUESTION_EXAMPLES["question_2"]},
        {"role": "assistant", "content": deception["example_2"]},
        {"role": "user", "content": QUESTION_EXAMPLES["question_3"]},
        {"role": "assistant", "content": deception["example_3"]},
        {"role": "user", "content": question},
    ]

def generate_response(messages, model, tokenizer, max_new_tokens):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def prepare_deceptive_non_falsities(df, model, tokenizer):
    result = df.copy()

    for deception_key in DECEPTION_TYPES:
        col_name = f"deceptive_{deception_key.replace(' ', '_')}"
        responses = []

        for question in tqdm(df["question"], desc=f"Generating '{deception_key}'"):
            messages = generate_prompt(question, deception_key)
            response = generate_response(messages, model, tokenizer)
            responses.append(response)

        result[col_name] = responses

    return result


def main():
    data_file = DATA_DIR / "true_false_dataset" / "true_false_dataset_with_questions.csv"
    df = pd.read_csv(data_file)

    model, tokenizer = load_model_and_tokenizer()
    df_deceptive_non_falsities = prepare_deceptive_non_falsities(df, model, tokenizer)
    df_deceptive_non_falsities.to_csv(DATA_DIR / "true_false_dataset" / "deceptive_non_falsities.csv", index=False)
    

if __name__ == "__main__":
    main()