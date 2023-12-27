
def create_dict(start_time, end_time, transcript, confidence_scores=None):
    # Split the transcript into words
    words = transcript.split()

    # Make the length of confidence scores equal to the length of words
    if confidence_scores is None:
        confidence_scores = [1.0] * (len(words))

    # Calculate the average confidence score for the sentence
    sentence_score = sum(confidence_scores) / len(confidence_scores)

    # Format time stamps
    time_stamp = [
        f"{int(start_time // 3600):0>2}:{int((start_time % 3600) // 60):0>2}:{start_time % 60:06.3f}".replace('.', ','),
        f"{int(end_time // 3600):0>2}:{int((end_time % 3600) // 60):0>2}:{end_time % 60:06.3f}".replace('.', ',')
    ]

    # Create the dictionary
    result_dict = {
        "0_time_stamp": time_stamp,
        "1_transcript": transcript,
        "2_confidence_score": confidence_scores,
        "3_sentence_score": sentence_score
    }

    return result_dict

# # Example usage
# start_time = 0.001
# end_time = 15.000
# transcript = "जी से निवेदन करता हूं कि हम सबका मार्गदर्शन करें इस कार्यक्रम में जुड़े अलग राज्यों के माननीय राज्यपाल श्री"
# confidence_scores = [
#     0.10994774848222733, 0.8041198253631592, 0.9727123379707336, 0.9845118522644043, 0.46442222595214844,
#     0.9817016124725342, 0.9773603677749634, 0.9378394484519958, 0.849056601524353, 0.6444385051727295,
#     0.9242770075798035, 0.9523084759712219, 0.9563120007514954, 0.9329007267951965, 0.6200013756752014,
#     0.9275965690612793, 0.9773687124252319, 0.8235010504722595, 0.9376240968704224, 0.12717460095882416
# ]

# result = create_dict(start_time, end_time, transcript, confidence_scores)
# print(result)
