from BiasPredictor import biasPredictor
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(file_path, model_path, explain = False):
	predictor = biasPredictor(model_path)
	pol_class, explanation = predictor.predict(file_path = file_path, explain = explain)

	print(f'Bias: {pol_class}')
	if explanation != None:
		key_phrases = "\n"+"\n".join(explanation)
		print(f'Key phrases: {key_phrases}')


if __name__=='__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-file_path',"--file_path", required=True, help="Path to the text file")
	ap.add_argument('-model_folder',"--model_folder", required=True, help="Math to the model")
	ap.add_argument('-explain',"--explain",type=str2bool, help="If model explanation needed")

	args = vars(ap.parse_args())
	file_path = args['file_path']
	model_path = args['model_folder']
	
	if 'explain' in args:
		explain = args['explain']
	else:
		explain = False
	main(file_path, model_path, explain)
