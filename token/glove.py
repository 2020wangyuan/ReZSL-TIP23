from transformers import AutoTokenizer, CLIPTextModel,CLIPTextModelWithProjection
import torch

model = CLIPTextModel.from_pretrained("/mnt/mydisk1/wangyuan/project/clip-vit-base-patch16")
tokenizer = AutoTokenizer.from_pretrained("/mnt/mydisk1/wangyuan/project/clip-vit-base-patch16")

inputs = tokenizer([
    "Colors",  # Colors: black, white, blue, brown, gray, orange, red, yellow
    "Patterns",  # Patterns: patches, spots, stripes
    "Skin/Fur",  # Skin/Fur: furry, hairless, toughskin
    "Size",  # Size: big, small
    "Body Type",  # Body Type: bulbous, lean
    "Limbs",  # Limbs: flippers, hands, hooves, pads, paws, longleg, longneck, tail
    "Teeth",  # Teeth: chewteeth, meatteeth, buckteeth, strainteeth
    "Horns/Claws",  # Horns/Claws: horns, claws, tusks
    "Smell",  # Smell: smelly
    "Movement",  # Movement: flys, hops, swims, tunnels, walks, fast, slow
    "Strength",  # Strength: strong, weak, muscle
    "Posture",  # Posture: bipedal, quadrapedal
    "Activity Level",  # Activity Level: active, inactive, nocturnal, hibernate, agility
    "Diet",  # Diet: fish, meat, plankton, vegetation, insects
    "Foraging Behavior",  # Foraging Behavior: forager, grazer, hunter, scavenger, skimmer, stalker
    "Geographic Origin",  # Geographic Origin: newworld, oldworld
    "Habitat",
    # Habitat: arctic, coastal, desert, bush, plains, forest, fields, jungle, mountains, ocean, ground, water, tree, cave
    "Temperament",  # Temperament: fierce, timid
    "Intelligence",  # Intelligence: smart
    "Social Behavior",  # Social Behavior: group, solitary
    "Nesting",  # Nesting: nestspot
    "Domesticity"  # Domesticity: domestic
], padding=True, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state

pooled_output = outputs.pooler_output  # pooled (EOS token) states
torch.save(pooled_output, 'awa_group_att_name_w2v.pt')


pass