import argparse
import csv
import datetime
import matlab.engine
import numpy as np
import os
import pandas as pd
import random
import soundfile as sf

from scipy.io import savemat
from scipy.signal import resample_poly

S_N_DIR = 's{}'
MIX_DIR = 'mix'

N_TR = 4000  # number of training files to generate
N_TT = 600   # number of test files to generate
N_CV = 1000  # number of validation files to generate

FILELIST_STUB = 'mix_{}_spk_filenames_{}.csv'


def read_wav(path, downsample_8k=False):
    samples, sr_orig = sf.read(path)
    if len(samples.shape) > 1:
        samples = samples[:, 0]

    if downsample_8k:
        samples = resample_poly(samples, 8000, sr_orig)

    return samples


def write_wav(file, samples, sr):
    int_samples = np.int16(np.round((2 ** 15) * samples))
    sf.write(file, int_samples, sr, subtype='PCM_16')


def fix_length(utterances, min_or_max='max'):
    if min_or_max == 'min':
        utt_len = np.min(np.array(list(map(len, utterances))))
    else:  # max
        utt_len = np.max(np.array(list(map(len, utterances))))
    for i in range(len(utterances)):
        if len(utterances[i]) >= utt_len:
            utterances[i] = utterances[i][:utt_len]
        else:
            utterances[i] = np.append(utterances[i], np.zeros(utt_len - len(utterances[i])))


def get_max_amplitude(mix, utterances):
    audio_tracks = utterances
    audio_tracks.append(mix)
    max_amplitude = 0
    for i in range(len(audio_tracks)):
        current_max = np.max(np.abs(audio_tracks[i]))
        if current_max > max_amplitude:
            max_amplitude = current_max

    return max_amplitude


def create_wsj0_mix_n_split_csv(wsj_root, csv_path, n_speakers, split, random_seed=0):
    random.seed(random_seed)
    np.random.seed(random_seed)

    wsj0_tr_speakers_list = [os.path.basename(f.path) for f in os.scandir(os.path.join(wsj_root, 'wsj0', 'si_tr_s')) if f.is_dir()]
    split_index = int(np.round(len(wsj0_tr_speakers_list) * (N_TR / (N_TR + N_CV))))

    wsj0_speakers_tr = wsj0_tr_speakers_list[:split_index]
    wsj0_speakers_cv = wsj0_tr_speakers_list[split_index:]

    wsj0_speakers_tt_dt = [os.path.basename(f.path) for f in os.scandir(os.path.join(wsj_root, 'wsj0', 'si_dt_05')) if f.is_dir()]
    wsj0_speakers_tt_et = [os.path.basename(f.path) for f in os.scandir(os.path.join(wsj_root, 'wsj0', 'si_et_05')) if f.is_dir()]

    if split == 'tt':
        n_mixtures = N_TT
        wsj0_speakers = wsj0_speakers_tt_dt + wsj0_speakers_tt_et
    elif split == 'cv':
        n_mixtures = N_CV
        wsj0_speakers = wsj0_speakers_cv
    else:
        n_mixtures = N_TR
        wsj0_speakers = wsj0_speakers_tr

    rows = []
    for i_mix in range(n_mixtures):
        amp_values_db = np.random.rand(n_speakers) * 10.0 - 5.0
        if n_speakers % 2 == 0:  # n even
            amp_values_db[1] = amp_values_db[0] * -1.0
        elif n_speakers % 2 == 1 and n_speakers > 1:  # n odd and n > 1
            amp_values_db[0] = 0.0
            amp_values_db[2] = amp_values_db[1] * -1.0

        random_speakers = random.sample(wsj0_speakers, k=n_speakers)
        random_utterances = []
        for speaker in random_speakers:
            if split == 'tt' and speaker in wsj0_speakers_tt_dt:
                file_path = os.path.join(wsj_root, 'wsj0', 'si_dt_05', speaker)
            elif split == 'tt' and speaker in wsj0_speakers_tt_et:
                file_path = os.path.join(wsj_root, 'wsj0', 'si_et_05', speaker)
            else:
                file_path = os.path.join(wsj_root, 'wsj0', 'si_tr_s', speaker)

            files = [f.path for f in os.scandir(file_path) if f.path.endswith('.wav')]
            random_file = random.choice(files)
            path_parts = []
            for _ in range(4):
                random_file = os.path.split(random_file)
                path_parts.append(random_file[1])
                random_file = random_file[0]

            random_utterance = os.path.join(*path_parts[::-1])
            random_utterances.append(random_utterance)

        output_filename = ''
        for i, file in enumerate(random_utterances):
            output_filename += os.path.split(file)[-1][:-4]
            output_filename += '_'
            output_filename += '{:.4f}'.format(amp_values_db[i])
            output_filename += '_'

        output_filename = output_filename[:-1] + '.wav'
        row_entry = [output_filename] + random_utterances + amp_values_db.tolist()
        rows.append(row_entry)

    fields = ['output_filename']
    for i in range(n_speakers):
        fields.append('s{}_path'.format(i + 1))

    for i in range(n_speakers):
        fields.append('s{}_snr'.format(i + 1))

    csv_filename = os.path.join(csv_path, FILELIST_STUB.format(n_speakers, split))
    with open(csv_filename, mode='w', newline='') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

    print('CSV file creation finished for the data split: {}'.format(split))


def main(wsj0_root, output_root, n_speakers, sr_str='8k', data_length='min'):
    assert n_speakers > 0, 'The number of speakers to mix must be greater than zero'
    assert sr_str in ['8k', '16k', 'both'], 'The sample rate argument must be one of: 8k / 16k / both'
    assert data_length in ['max', 'min', 'both'], 'The data length argument must be one of: min / max / both'

    mlab_engine = matlab.engine.start_matlab()
    buffer_file = os.path.join(output_root, 'wsj0-mix{}'.format(n_speakers), 'helper.mat')

    wav_path = os.path.join(output_root, 'wsj0-mix{}'.format(n_speakers))
    csv_path = os.path.join(wav_path, 'data')
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    for split in ['tr', 'cv', 'tt']:
        create_wsj0_mix_n_split_csv(wsj0_root, csv_path, n_speakers, split)

    if sr_str == 'both':
        sr_options = [8000, 16000]
    elif sr_str == '16k':
        sr_options = [16000]
    else:
        sr_options = [8000]

    for sr in sr_options:
        wav_dir = 'wav' + '{}k'.format(int(sr / 1000))
        if sr == 8000:
            downsample = True
        else:
            downsample = False

        if data_length == 'both':
            datalen_options = ['max', 'min']
        elif data_length == 'max':
            datalen_options = ['max']
        else:
            datalen_options = ['min']

        for datalen in datalen_options:
            for split in ['tr', 'cv', 'tt']:
                output_path = os.path.join(wav_path, wav_dir, datalen, split)

                for i in range(n_speakers):
                    speaker_i_output_dir = os.path.join(output_path, S_N_DIR.format(i + 1))
                    os.makedirs(speaker_i_output_dir, exist_ok=True)

                mix_output_dir = os.path.join(output_path, MIX_DIR)
                os.makedirs(mix_output_dir, exist_ok=True)

                print('{} {} dataset, {} split'.format(wav_dir, datalen, split))

                # read filenames
                wsjmix_path = os.path.join(csv_path, FILELIST_STUB.format(n_speakers, split))
                wsjmix_df = pd.read_csv(wsjmix_path)

                for i_utt, csv_tuple in enumerate(wsjmix_df.itertuples(index=False, name=None)):
                    scalings = []
                    utterances = []
                    for i in range(n_speakers):
                        s_i = read_wav(os.path.join(wsj0_root, csv_tuple[i + 1]), downsample)

                        # hack to significantly increase the performance of passing python arrays to the matlab engine
                        savemat(buffer_file, dict(s_i=s_i))

                        # active speech level computation in units of power
                        level_i = mlab_engine.activlev(mlab_engine.load(buffer_file)['s_i'], mlab_engine.double(sr), 'n', nargout=2)[1]
                        scaling_i = 1 / np.sqrt(level_i)  # active speech level normalization to 0 dB

                        # application of the target SNR (converted from dB to linear scale)
                        scaling_i *= 10 ** (csv_tuple[i + n_speakers + 1] / 20)
                        s_i *= scaling_i
                        scalings.append(scaling_i)
                        utterances.append(s_i)

                    fix_length(utterances, datalen)
                    mix = np.sum(np.array(utterances), axis=0)

                    output_name = csv_tuple[0]
                    max_amplitude = get_max_amplitude(mix, utterances)
                    max_peak_norm = 0.9 * (1 / max_amplitude)  # maximum peak normalization factor
                    for i in range(n_speakers):
                        speaker_i_output_dir = os.path.join(output_path, S_N_DIR.format(i + 1))
                        write_wav(os.path.join(speaker_i_output_dir, output_name), utterances[i] * max_peak_norm, sr)

                    write_wav(os.path.join(mix_output_dir, output_name), mix * max_peak_norm, sr)

                    if (i_utt + 1) % 100 == 0:
                        print('Completed {} of {} utterances  --  {}'.format(i_utt + 1, len(wsjmix_df), str(datetime.datetime.now())))

    # remove the matlab helper file when finished
    if os.path.exists(buffer_file):
        os.remove(buffer_file)

    mlab_engine.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, help='Output directory for writing wsj0-mix-n 8 kHz and 16 kHz datasets')
    parser.add_argument('--wsj0-root', type=str, help='Path to the folder containing wsj0/')
    parser.add_argument('--sr-str', type=str, help='The target sample rate of the created wav files. Choose one: 8k / 16k / both')
    parser.add_argument('--data-length', type=str,
                        help='Whether to use the maximum or minimum length of the selected utterances. Choose one: min / max / both')
    parser.add_argument('--n', type=int, help='Number of speakers to mix in each mixture')
    args = parser.parse_args()
    main(args.wsj0_root, args.output_dir, args.n, args.sr_str, args.data_length)
