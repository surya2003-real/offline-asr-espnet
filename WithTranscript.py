import jiwer
def with_transcript(transcript, asr_prediction, tol = 6):
    hyp = jiwer.ReduceToListOfListOfWords()(transcript)
    tru = jiwer.ReduceToListOfListOfWords()(asr_prediction)
    out = jiwer.process_words(asr_prediction, transcript)
    req = out.alignments[0]
    # print(req)
    text_out = []
    for it in req:
        if it.type == 'delete':
            for i in range(it.ref_start_idx, it.ref_end_idx):
                if tru[0][i] == '|':
                    text_out.append('|')
                    continue
                if it.ref_end_idx - it.ref_start_idx > tol:
                    text_out.append(tru[0][i])
        elif it.type == 'substitute':
            idx_diff = it.ref_start_idx-it.hyp_start_idx
            for i in range(it.ref_start_idx, it.ref_end_idx):
                if tru[0][i] == '|':
                    text_out.append('|')
                    text_out.append(hyp[0][i-idx_diff])
                else:
                    text_out.append(hyp[0][i-idx_diff])
        else:
            text_out = text_out + hyp[0][it.hyp_start_idx:it.hyp_end_idx]
    sentence = ' '.join(text_out)
    sentence_list = sentence.split(' | ')
    sentence_list = [sentence.strip() for sentence in sentence_list if sentence.strip()]
    return sentence_list

def solo_label_2(asr_dict):
    asr_trans=[d.get('1_transcript') for d in asr_dict]
    asr = ''
    for label in asr_trans:
        if asr == '':
            asr = label
        else:
            asr = asr+' | '+label
    return asr
# out1 = with_transcript(text_curr, text2, 5)
# out1