# =========================================================
# CALORIE DATA
# Approximate calories per 100 grams
# =========================================================

CALORIE_DATA = {

    "apple_pie": 237,
    "baby_back_ribs": 320,
    "baklava": 334,
    "beef_carpaccio": 120,
    "beef_tartare": 180,

    "beet_salad": 70,
    "beignets": 300,
    "bibimbap": 140,
    "bread_pudding": 230,
    "breakfast_burrito": 180,

    "bruschetta": 150,
    "caesar_salad": 160,
    "cannoli": 270,
    "caprese_salad": 180,
    "carrot_cake": 415,

    "ceviche": 90,
    "cheesecake": 320,
    "cheese_plate": 350,
    "chicken_curry": 190,
    "chicken_quesadilla": 240,

    "chicken_wings": 290,
    "chocolate_cake": 370,
    "chocolate_mousse": 225,
    "churros": 350,
    "clam_chowder": 95,

    "club_sandwich": 250,
    "crab_cakes": 220,
    "creme_brulee": 330,
    "croque_madame": 260,
    "cup_cakes": 305,

    "deviled_eggs": 145,
    "donuts": 400,
    "dumplings": 220,
    "edamame": 120,
    "eggs_benedict": 220,

    "escargots": 150,
    "falafel": 330,
    "filet_mignon": 270,
    "fish_and_chips": 230,
    "foie_gras": 462,

    "french_fries": 312,
    "french_onion_soup": 90,
    "french_toast": 230,
    "fried_calamari": 175,
    "fried_rice": 175,

    "frozen_yogurt": 127,
    "garlic_bread": 350,
    "gnocchi": 150,
    "greek_salad": 120,
    "grilled_cheese_sandwich": 350,

    "grilled_salmon": 206,
    "guacamole": 150,
    "gyoza": 200,
    "hamburger": 295,
    "hot_and_sour_soup": 60,

    "hot_dog": 290,
    "huevos_rancheros": 160,
    "hummus": 166,
    "ice_cream": 207,
    "lasagna": 135,

    "lobster_bisque": 95,
    "lobster_roll_sandwich": 250,
    "macaroni_and_cheese": 164,
    "macarons": 450,
    "miso_soup": 40,

    "mussels": 172,
    "nachos": 300,
    "omelette": 154,
    "onion_rings": 411,
    "oysters": 81,

    "pad_thai": 150,
    "paella": 180,
    "pancakes": 227,
    "panna_cotta": 220,
    "peking_duck": 337,

    "pho": 60,
    "pizza": 266,
    "pork_chop": 231,
    "poutine": 250,
    "prime_rib": 300,

    "pulled_pork_sandwich": 250,
    "ramen": 80,
    "ravioli": 180,
    "red_velvet_cake": 367,
    "risotto": 166,

    "samosa": 262,
    "sashimi": 150,
    "scallops": 111,
    "seaweed_salad": 70,
    "shrimp_and_grits": 150,

    "spaghetti_bolognese": 160,
    "spaghetti_carbonara": 190,
    "spring_rolls": 150,
    "steak": 271,
    "strawberry_shortcake": 220,

    "sushi": 150,
    "tacos": 200,
    "takoyaki": 210,
    "tiramisu": 240,
    "tuna_tartare": 130,

    "waffles": 291
}


# =========================================================
# FUNCTION
# =========================================================

def get_calorie_info(food_name):

    return CALORIE_DATA.get(
        food_name,
        0
    )