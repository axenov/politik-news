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

def main(file_path, method, explain = False):
	predictor = biasPredictor(method)
	prediction = predictor.predict(file_path = file_path, explain = explain)

	if explain:
		print(f'Bias: {prediction[0]}')
		key_phrases = "\n"+"\n".join(prediction[1])
		print(f'Key phrases: {key_phrases}')
	else:
		print(f'Bias: {prediction}')



if __name__=='__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-file_path',"--file_path", required=True, help="Path to the text file")
	ap.add_argument('-method',"--method", required=True, help="Features used by the model (bert or tfidf)")
	ap.add_argument('-explain',"--explain",type=str2bool, help="Set if model explanation is needed")

	args = vars(ap.parse_args())
	file_path = args['file_path']
	method = args['method']
	
	if 'explain' in args:
		explain = args['explain']
	else:
		explain = False
	main(file_path, method, explain)
