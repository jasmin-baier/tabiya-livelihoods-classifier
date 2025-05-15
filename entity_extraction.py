
from inference.linker import EntityLinker
pipeline = EntityLinker(k=5)
text = 'We are looking for a Head Chef who can plan menus.'
extracted = pipeline(text)
print(extracted)