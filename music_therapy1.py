from __future__ import print_function
from __future__ import division
from music21 import *
from collections import defaultdict, OrderedDict
from itertools import groupby, zip_longest
from grammer import *
import skfuzzy as fuzz 
#----------------------------HELPER FUNCTIONS----------------------------------#
 
''' Helper function to parse a MIDI file into its measures and chords '''
def __parse_midi(data_fn):
    # Parse the MIDI data for separate melody and accompaniment parts.
    midi_data = converter.parse(data_fn)
    # Get melody part, compress into single voice.
    melody_stream = midi_data[5]
    # For Metheny piece, Melody is Part #5.
    melody1,melody2 = melody_stream.getElementsByClass(stream.Voice)
    for j in melody2:
        melody1.insert(j.offset, j)
    melody_voice = melody1
    for i in melody_voice:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25
 
    # Change key signature to adhere to comp_stream (1 sharp, mode = major).
    # Also add Electric Guitar. 
    melody_voice.insert(0, instrument.ElectricGuitar())     
 
    # The accompaniment parts. Take only the best subset of parts from
    # the original data. Maybe add more parts, hand-add valid instruments.
    # Should add least add a string part (for sparse solos).
    # Verified are good parts: 0, 1, 6, 7 '''
    partIndices = [0, 1, 6, 7]
    comp_stream = stream.Voice()
    comp_stream.append([j.flat for i, j in enumerate(midi_data) 
        if i in partIndices])
 
    # Full stream containing both the melody and the accompaniment. 
    # All parts are flattened. 
    full_stream = stream.Part()
    for i in range(len(comp_stream)):
        full_stream.append(comp_stream[i])
    full_stream.append(melody_voice)
 
    # Extract solo stream, assuming you know the positions ..ByOffset(i, j).
    # Note that for different instruments (with stream.flat), you NEED to use
    # stream.Part(), not stream.Voice().
    # Accompanied solo is in range [478, 548)
    solo_stream = stream.Part()
    for part in full_stream:
        curr_part = stream.Part()
        curr_part.append(part.getElementsByClass(instrument.Instrument))
        curr_part.append(part.getElementsByClass(tempo.MetronomeMark))
        curr_part.append(part.getElementsByClass(key.KeySignature))
        curr_part.append(part.getElementsByClass(meter.TimeSignature))
        curr_part.append(part.getElementsByOffset(476, 548, 
                                                  includeEndBoundary=True))
        cp = curr_part.flat
        solo_stream.insert(cp)
 
    # Group by measure so you can classify. 
    # Note that measure 0 is for the time signature, metronome, etc. which have
    # an offset of 0.0.
    melody_stream = solo_stream[-1]
    measures = OrderedDict()
    offsetTuples = [(int(n.offset / 4), n) for n in melody_stream]
    measureNum = 0 # for now, don't use real m. nums (119, 120)
    for key_x, group in groupby(offsetTuples, lambda x: x[0]):
        measures[measureNum] = [n[1] for n in group]
        measureNum += 1  
 
    # Get the stream of chords.
    # offsetTuples_chords: group chords by measure number.
    chordStream = solo_stream[0]
    chordStream.removeByClass(note.Rest)
    chordStream.removeByClass(note.Note)
    offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]
 
    # Generate the chord structure. Use just track 1 (piano) since it is
    # the only instrument that has chords. 
    # Group into 4s, just like before. 
    chords = OrderedDict()
    measureNum = 0
    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
        chords[measureNum] = [n[1] for n in group]
        measureNum += 1
 
    # Fix for the below problem.
    #   1) Find out why len(measures) != len(chords).
    #   ANSWER: resolves at end but melody ends 1/16 before last measure so doesn't
    #           actually show up, while the accompaniment's beat 1 right after does.
    #           Actually on second thought: melody/comp start on Ab, and resolve to
    #           the same key (Ab) so could actually just cut out last measure to loop.
    #           Decided: just cut out the last measure. 
    del chords[len(chords) - 1]
    assert len(chords) == len(measures)
    return measures, chords
 
''' Helper function to get the grammatical data from given musical data. '''
def __get_abstract_grammars(ma, ca):
    # extract grammars
    abstract_grammars = []
    for ix in range(1, len(ma)):
        m = stream.Voice()
        for i in ma[ix]:
            m.insert(i.offset, i)
        c = stream.Voice()
        for j in ca[ix]:
            c.insert(j.offset, j)
        print(m,c)
        parsed = parse_melody(m, c)
        abstract_grammars.append(parsed)
 
    return abstract_grammars
 
#----------------------------PUBLIC FUNCTIONS----------------------------------#
 
''' Get musical data from a MIDI file '''
def get_musical_data(data_fn):
    measures, chords = __parse_midi(data_fn)
    abstract_grammars = __get_abstract_grammars(measures, chords)
 
    return chords, abstract_grammars
def get_corpus_data(abstract_grammars):
    corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]
    values = set(corpus)
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))
 
    return corpus, values, val_indices, indices_val
def __sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
 
 
# In[6]:
 
''' Helper function to generate a predicted value from a given matrix '''
def __predict(model, x, indices_val, diversity):
    print("------------0000000>>>>>"+str(x.shape))
    preds = model.predict(x, verbose=0)[0]
    next_index = __sample(preds, diversity)
    next_val = indices_val[next_index]
 
    return next_val
def __generate_grammar(model, corpus, abstract_grammars, values, val_indices,
                       indices_val, max_len, max_tries, diversity):
    curr_grammar = ''
    # np.random.randint is exclusive to high
    start_index = np.random.randint(0, abs(len(corpus) - max_len))
    sentence = corpus[start_index: start_index + max_len]
    # seed
    print(sentence)
    running_length = 0.0
    print(val_indices)
    z=list(val_indices)
    z=np.array([[ord(i.split(",")[0]),float(i.split(",")[1])] for i in z])
    while running_length <= 4.1:    # arbitrary, from avg in input file
        # transform sentence (previous sequence) to matrix
        x = np.zeros((1, 1, len(values[0])))
        print(type(x))
        for t,val in enumerate(sentence):
            print(t,val)
            try:
                print(str(str(chr(int(val[0][0])))+","+str(val[0][1])))
            except:
                pass
            if (not val in z): print(val)
            if(t<1):
                try:
                    x[0, t, val_indices[str(str(chr(int(val[0][0])))+","+str(val[0][1])+"00")]] = 1
                except:
                    pass
                
 
        next_val = __predict(model, x, indices_val, diversity)
 
        # fix first note: must not have < > and not be a rest
        if (running_length < 0.00001):
            tries = 0
            while (next_val.split(',')[0] == ord('R') or 
                len(next_val.split(',')) != 2):
                # give up after 1000 tries; random from input's first notes
                if tries >= max_tries:
                    print('Gave up on first note generation after', max_tries, 
                        'tries')
                    # np.random is exclusive to high
                    rand = np.random.randint(0, len(abstract_grammars))
                    next_val = abstract_grammars[rand].split(' ')[0]
                else:
                    next_val = __predict(model, x, indices_val, diversity)
 
                tries += 1
 
        # shift sentence over with new value
        sentence = list(sentence[1:])
        
        sentence.append(next_val)
 
        # except for first case, add a ' ' separator
        if (running_length > 0.00001): curr_grammar += ' '
        curr_grammar += next_val
 
        length = float(next_val.split(',')[1])
        running_length += length
    return curr_grammar
from preprocess import *
from keras import *
from keras.layers import *
import numpy as np
from qa import *
data=r"C:\\Python35\\Pop_Music_Midi\\aashiqui_ab_teray_bin_jee_lenge_hum.mid"
chords,abstr=get_musical_data(data)
x,y,z,w=get_corpus_data(abstr)
model=Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(1,2)))
model.add(Dropout(0.2))
model.add(LSTM(128,return_sequences=False,input_shape=(20,2)))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
x=np.array([[ord(i.split(",")[0]),float(i.split(",")[1])] for i in x])
y=list(y)
y=np.array([[ord(i.split(",")[0]),float(i.split(",")[1])] for i in y])
x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
y=np.resize(y,(19,2))
model.fit(x,y,batch_size=128,nb_epoch=128)
max_len=20
max_tries=1000
from io import StringIO
diversity=0.5
out_stream = stream.Stream()
curr_grammar=__generate_grammar(model,x,abstr,y,z,w,max_len,max_tries,diversity)
loopEnd = len(chords)
from math import *
import librosa as li
import numpy as np 
from sklearn.cluster import AffinityPropagation, KMeans
import matplotlib.pyplot as plt
from scipy import stats
from math import *
import operator
def prob(arr):
    l=list(arr)
    di={}
    for x in range(len(l)):
        di[l[x]]=l.count(l[x])
    sorted(di.items(), key=operator.itemgetter(1))
    return di.keys()[0]
file_name = r"C:\\Python27\\Angry\\Angry1.wav"
audio_time_series, sample_rate = li.load(file_name)
length_series = len(audio_time_series)
print(length_series)
zero_crossings = []
energy = []
entropy_of_energy = []
mfcc = []
chroma_stft = []
for i in range(0,length_series,int(sample_rate/2)):
     frame_self = audio_time_series[i:i+int(sample_rate/2):1]
     z = li.zero_crossings(frame_self)
     arr = np.nonzero(z)
     zero_crossings.append(len(arr[0]))
     e = li.feature.rmse(frame_self)
     energy.append(np.mean(e))
     ent = 0.0
     m = np.mean(e)
     for j in range(0,len(e[0])):
          q = np.absolute(e[0][j] - m)
          ent = ent + (q * np.log10(q))
     entropy_of_energy.append(ent)
     mt = []
     mf = li.feature.mfcc(frame_self)
     for k in range(0,len(mf)):
          mt.append(np.mean(mf[k]))
     mfcc.append(mt)
     ct = []
     cf = li.feature.chroma_stft(frame_self)
     for k in range(0,len(cf)):
          ct.append(np.mean(cf[k]))
     chroma_stft.append(ct)
f_list_1 = []
f_list_1.append(zero_crossings)
f_list_1.append(energy)
f_np_1 = np.array(f_list_1)
f_np_1 = np.transpose(f_np_1)
sp_centroid = []
sp_bandwidth = []
sp_contrast = []
sp_rolloff = []
for i in range(0,length_series,int(sample_rate/2)):
     frame_self = audio_time_series[i:i+int(sample_rate/2):1]
     cp = li.feature.spectral_centroid(y=frame_self, hop_length=220500)
     sp_centroid.append(cp[0][0])
     pitches, magnitudes = li.core.piptrack(y=frame_self, sr=sample_rate, S=None, hop_length=220500, fmin=80, fmax=250, threshold=0.75)
     sp_bandwidth.append(pitches[0][0])
     csp = li.feature.spectral_contrast(y=frame_self, hop_length=220500)
     sp_contrast.append(np.mean(csp))
     rsp = li.onset.onset_strength(y=frame_self, sr=sample_rate)
     sp_rolloff.append(np.mean(rsp[0]))
     

f_list_2 = []
f_list_2.append(sp_bandwidth)

f_np_2 = np.array(f_list_2)
f_np_2 = np.transpose(f_np_2)

f_np_3 = np.array(mfcc)

master = np.concatenate([f_np_1, f_np_2, f_np_3], axis=1)

#cluster_obj = AffinityPropagation().fit(master)
cluster_obj = KMeans(n_clusters = 3 ,random_state=0).fit(master)
#print("Number of clusters : " + str(len(cluster_obj.cluster_centers_indices_)))
res = cluster_obj.predict(master)
#print(cluster_obj.get_params())
s = res[0]
index={0:"Senior Citizen",1:"Young",2:"Child"}
avg=sum(res)/len(res)
a=np.bincount(res)
if(avg-int(avg)>=0.5):
    print(index[np.argmax(a)])
else:
    print(index[np.argmax(a)])


#emotion detection

f_list_2 = []
f_list_2.append(sp_rolloff)
f_np_2 = np.array(f_list_2)
f_np_2 = np.transpose(f_np_2)

f_np_3 = np.array(mfcc)
master = np.concatenate([f_np_1, f_np_2, f_np_3], axis=1)
cluster_obj = KMeans(n_clusters = 4 ,random_state=0).fit(master)
res = cluster_obj.predict(master)
s = res[0]
avg=sum(res)/len(res)
index={0:"Angry",1:"Normal",2:"Sad",3:"Happy"}
a=np.bincount(res)
if(avg-int(avg)>=0.5):
    print(index[np.argmax(a)])
else:
    print(index[np.argmax(a)])
def slope(ax,ay):
    return ay/ax
def angle(m1,m2):
    return atan(abs(m1-m2)/(1+m1*m2))
x=0
plt.plot([0,1,2,3,4,5],linestyle='-')
if(np.argmax(a)==0):
    res=list(res)
    angx=res.count(0)*(res.count(0)/len(res))
    angy=res.count(2)*(res.count(2)/len(res))
    plt.plot([angx+1,angy],[angx+2,angy],linestyle='-')
    if(angle(slope(angx,angy),1)>0):
        print("Actively angry")
    else:
        print("less actively angry")
if(np.argmax(a)==3):
    res=list(res)
    angx=res.count(3)*(res.count(3)/len(res))
    angy=res.count(2)*(res.count(2)/len(res))
    print(angx,angy)
    if(angle(slope(angx,angy),1)>0):
        print("Actively happy")
    else:
        print("less actively happy")
if(np.argmax(a)==2):
    res=list(res)
    angx=res.count(1)*(res.count(2)/len(res))
    angy=res.count(1)*(res.count(1)/len(res))
    print (angx,angy)
    if(angle(slope(angx,angy),1)>0):
        print("Passively sad")
    else:
        print("less passively angry")
plt.xlabel("Angry")
plt.ylabel("Sad")
plt.show()
bpm=100
if(np.argmax(a)==0):
    bpm=150
if(np.argmax(a)==3):
    bpm=100
if(np.argmax(a)==2):
    bpm=50
curr_offset=0.0
for loopIndex in range(1, loopEnd):
        # get chords from file
    curr_chords = stream.Voice()
    for j in chords[loopIndex]:
        curr_chords.insert((j.offset), j)

        # generate grammar


        # Pruning #1: smoothing measure
    curr_grammar = prune_grammar(curr_grammar)

        # Get notes from grammar and chords
    curr_notes = unparse_grammar(curr_grammar, curr_chords)

        # Pruning #2: removing repeated and too close together notes
    curr_notes = prune_notes(curr_notes)

        # quality assurance: clean up notes
    curr_notes = clean_up_notes(curr_notes)

        # print # of notes in curr_notes
    print('After pruning: %s notes' % (len([i for i in curr_notes
                                            if isinstance(i, note.Note)])))

        # insert into the output stream
    for m in curr_notes:
        out_stream.insert(curr_offset + m.offset, m)
    for mc in curr_chords:
        out_stream.insert(curr_offset + mc.offset, mc)
    curr_offset += 5.0

out_stream.insert(0.0, tempo.MetronomeMark(number=bpm))

    # Play the final stream through output (see 'play' lambda function above)
    
    # save stream
mf=midi.translate.streamToMidiFile(out_stream)
mf.open("aks.midi","wb")
mf.write()
mf.close()
graph.plot.ScatterPitchClassQuarterLength(out_stream).run()
