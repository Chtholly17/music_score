import base64
import os
import json
import random
import main_file
import numpy as np
import converter
import featureExtract as fe
# pp = Flask(__name__)
#  CORS(app, supports_credentials=True)# 
# app.route('/score', methods=["POST", "GET"])
# ef api_score():
#    if request.method == 'POST':# 
#        sung_song_path = request.form['sung_song_path']
#        reference_song_path = request.form['reference_song_path']
#        begin_time = request.form['begintime']
#        end_time = request.form['endtime']# 
#        print(sung_song_path)
#        print(reference_song_path)
#        print(begin_time)
#        print(end_time)# 
#        if not os.path.exists(sung_song_path) or not os.path.exists(reference_song_path):
#            return Response(json.dumps({'result': "origin music file not found!"}), content_type='application/json')# 
#        os.system('cp ' + sung_song_path + ' ./sung.wav')
#        os.system('cp ' + reference_song_path + ' ./ref.mp3')
#        # converter.trans_mp3_to_wav('./sung.mp3', './sung.wav')
#        converter.get_second_part_from_mp3_to_wav('./ref.mp3', float(begin_time), float(end_time), 'ref.wav')
#        scores = np.array(main_file.getFeatures('sung.wav', 'ref.wav'))
#        score = np.mean(scores)
#        print("score : ", score)
#        return Response(json.dumps({'result': "success!", 'score': score}), content_type='application/json')
#    else:
#        return Response(json.dumps({'response': "this is a get."}), content_type='application/json')# 

if __name__ == '__main__':
    converter.get_second_part_from_mp3_to_wav('1.mp3', 1.001, 10.001, '1.wav')
    converter.get_second_part_from_mp3_to_wav('2.mp3', 1.001, 10.001, '2.wav')
    # scores = np.array(main_file.getFeatures('1.wav', '2.wav'))
    # scores = np.mean(scores)
    featureExtract = fe.featureExtract()   
    featureExtract.forward_once('1.wav', '2.wav')