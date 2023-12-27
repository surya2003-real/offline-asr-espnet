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
            if it.ref_end_idx - it.ref_start_idx > tol:
                text_out = text_out + tru[0][it.ref_start_idx:it.ref_end_idx]
        else:
            text_out = text_out + hyp[0][it.hyp_start_idx:it.hyp_end_idx]
    sentence = ' '.join(text_out)
    return sentence

# out1 = with_transcript(text_curr, text2, 5)
# out1