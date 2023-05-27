from pydub import AudioSegment


def trans_mp3_to_wav(origin_filepath, target_filepath):
    song = AudioSegment.from_mp3(origin_filepath)
    song.export(target_filepath, format="wav")


def trans_wav_to_mp3(origin_filepath, target_filepath):
    song = AudioSegment.from_wav(origin_filepath)
    song.export(target_filepath, format="mp3")


def get_second_part_from_mp3_to_wav(main_mp3_path, start_time, end_time, part_wav_path):
    print(main_mp3_path)
    print(part_wav_path)
    """
    锟斤拷频锟斤拷片锟斤拷锟斤拷取锟斤拷锟斤拷锟斤拷频锟斤拷锟斤拷位锟斤拷
    :param main_mp3_path: 原锟斤拷频锟侥硷拷路锟斤拷
    :param start_time: 锟斤拷取锟侥匡拷始时锟斤拷
    :param end_time: 锟斤拷取锟侥斤拷锟斤拷时锟斤拷
    :param part_wav_path: 锟斤拷取锟斤拷锟斤拷锟狡德凤拷锟�
    :return:
    """
    start_time = int(start_time * 1000)
    end_time = int(end_time * 1000)

    sound = AudioSegment.from_mp3(main_mp3_path)
    word = sound[start_time:end_time]

    word.export(part_wav_path, format="wav")
