import pickle

def load_obj(name):
    with open('saved_things/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def save_obj(obj, name):
    with open('saved_things/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

taste = ["Semi-Dry", "SemiDry", "Semi Dry", "Medium-Dry", "MediumDry",  "Medium Dry", "Dry", \
         "Semi-Sweet", "SemiSweet", "Semi Sweet", "Sweet", "Medium-Sweet", "MediumSweet", "Medium Sweet", \
         "Full-Bodied", "FullBodied", "Full Bodied", "Medium-Bodied", "MediumBodied", "Medium Bodied", \
         "Crisp", "Balance", 'Sparkling', 'rough', 'tannic', 'tannins', 'smooth']
taste = [x.lower() for x in taste]

fruit = ["Honey", "Fruit", "Strawberry", "Raspberry", "Apple", "Citrus", "Vanilla", "Pineapple", "Sage", "Flower", \
         "Pear", "Mint", "Plum", "Blackberry", "Cherry", "Melon", "Peach", "Lemon", "Lime", "Mango", "Berry"]
fruit = [x.lower() for x in fruit]

with open("../knowledge/list_aromas.txt", "rb") as fp:
    aromas = pickle.load(fp)
aromas1 = aromas + ['spices', 'floral', 'cork', 'ripe', 'grabby','cigar', 'guava', 'ripe', \
                 'blackberry', 'herb', 'spice', 'fresh','rind', \
                'spice', 'espresso', 'buttercream','mineral','pepper','clove', \
                'licorice', 'tannins']
aromas3 = ['red berry fruits', 'dark plum fruit', 'oak-driven aromas']
aromas2 = ['berry fruits', 'black fruit', 'red fruit', 'black cherry', 'supple plum', 'green apple', 'red apple', \
            'dried fruit','dried sage','tropical fruit', 'baked plum', 'dark plum', 'candied berry', 'off dry', \
            'yellow flower', 'yellow fruit', 'yellow-fruit', 'white flower', 'savory herb', 'ripe pineapple',\
            'coffee beans', 'savory herb', 'white pepper', 'tannic backbone', 'coffee bean', 'apple notes' ,\
            'orange blossom']
aromas4 = ['juicy red berry fruits']
taste = ['finish', 'palate', 'acidity']

