
from inference.linker import EntityLinker
pipeline = EntityLinker(k=5, output_format='uuid')
text = 'We are looking for a Head Chef who can plan menus.'
extracted = pipeline(text)
print(extracted)

from inference.evaluator import Evaluator

results = Evaluator(entity_type='Skill', entity_model='tabiya/roberta-base-job-ner', similarity_model='all-MiniLM-L6-v2', crf=False, evaluation_mode=True)
print(results.output)