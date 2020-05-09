from scipy.io import wavfile
import os
import numpy as np
import argparse
from tqdm import tqdm

def windows(signal, step_size):
    for i_start in xrange(0, len(signal), step_size):
        i_end = i_start + step_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]

def energy(samples):
    return np.sum(np.power(samples, 2.)) / float(len(samples))

def remove_inside_silence_audio(input_filename, cut_minimum_duration, replace_duration, chunk_size, silence_threshold, output_dir, dry_run):
    print input_filename
    output_filename_prefix = os.path.splitext(os.path.basename(input_filename))[0]
    #read audio file and cut into chunks
    sample_rate, samples = input_data=wavfile.read(filename=input_filename, mmap=True)
    step_size = int(chunk_size * sample_rate)
    signal_windows = windows(signal=samples, step_size=step_size)

    #calculate energy for every chunk
    max_amplitude = np.iinfo(samples.dtype).max
    max_energy = energy([max_amplitude])
    window_energy = (energy(w) / max_energy for w in tqdm(signal_windows, total=int(len(samples) / float(step_size))))
    
    #make list to manipulate
    window_silence = []
    for e in window_energy:
        window_silence.append(e > silence_threshold)

    are_we_counting = False
    count_started = 0
    cut_slices = []
   
    for i, isHighEnergy in enumerate(window_silence):
        if i<2:
            continue
        if not isHighEnergy and not window_silence[i-1] and not window_silence[i-2]:
            are_we_counting = True
            if count_started==0:
                count_started = i-2
        if isHighEnergy and window_silence[i-1] and window_silence[i-2]:
            if are_we_counting and cut_minimum_duration<=(i-2-count_started)*chunk_size:
                cut_slices.append([count_started, i])
                #You play with this slice points a bit.
            are_we_counting = False
            count_started = 0
    
    #Appending the last part        
    if are_we_counting and cut_minimum_duration<=(i-2-count_started)*chunk_size:
        cut_slices.append([count_started, -1])
    elif len(cut_slices)>0:
        cut_slices.append([-1, -1])
       
    final_sample = None
    start_point = 0
    next_start_point = 0
        
    for i, entity in enumerate(cut_slices):
        if entity[0]==0:
            #Trimming from start
            start_point = entity[1]
            continue
            
        if next_start_point>0:
            start_point = next_start_point
        next_start_point = entity[1]
        end_point = entity[0]
        
        st = start_point*step_size
        ed = end_point*step_size
        if end_point==-1:
            ed = -1
        
        print "Making slice from: {} to: {}".format(start_point*chunk_size, end_point*chunk_size)
        
        this_slice = samples[st:ed]

        if final_sample is None:
            final_sample = this_slice
        else:
            final_sample = np.concatenate((final_sample,this_slice))
        
        if (entity[0]!=-1):
            #Replacing silence
            add_this_silence = int(replace_duration * sample_rate)
            final_sample = np.append(final_sample, [0.]*add_this_silence)
        
    if not dry_run and len(cut_slices)>0:
        final_sample = final_sample.astype(dtype=np.int16)
        output_final_path = "{}.wav".format(os.path.join(output_dir, output_filename_prefix))
        wavfile.write(filename=output_final_path, rate=sample_rate, data = final_sample)

if __name__ == "__main__":
    """
    usage
    python preprocess_inside_silence.py filelists/your_test_filelist.txt
    """
    parser = argparse.ArgumentParser(description='Remove inside silence from wav files')
    parser.add_argument('dataset_file', type=str, help='The text file containing WAV files to manipulate.')
    parser.add_argument('--output-dir', '-o', type=str, default='./cleared_output', help='The output folder. Defaults to the current folder. It doesnt replace actual files, you decide')
    parser.add_argument('--min-silence-length', '-m', type=float, default=0.5, help='Search for minimum duration to detect')
    parser.add_argument('--chunk-size', '-ch', type=float, default=0.05, help='Wav will be cut into small chunks to calculate')
    parser.add_argument('--silence-threshold', '-t', type=float, default=1e-3, help='The energy level (between 0.0 and 1.0) below which the signal is regarded as silent. You must tune it!')
    parser.add_argument('--replace-duration', '-s', type=float, default=0.4, help='Add standart silence duration to replace with silence')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Don\'t actually write any output files. Just report silence')

    args = parser.parse_args()
    dataset_file = args.dataset_file
    output_dir = args.output_dir
    cut_minimum_duration = args.min_silence_length
    chunk_size = args.chunk_size
    silence_threshold = args.silence_threshold
    replace_duration = args.replace_duration
    dry_run = args.dry_run
    f = open(dataset_file)
    R = f.readlines()
    f.close()
    for line in R:
        input_filename = line.split('|')[0]
        remove_inside_silence_audio(input_filename, cut_minimum_duration, replace_duration, chunk_size, silence_threshold, output_dir, dry_run)
