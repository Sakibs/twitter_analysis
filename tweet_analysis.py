import os
import json
import tarfile
import datetime, time


def get_num_tweets_hr(hashtag):
	filename = 'tweets_'+hashtag+'.txt'
	filepath = os.path.join('.', 'data', filename)
	f = open(filepath, 'r')

	out = open(os.path.join('.', 'part1', 'twts_hr_'+hashtag+'.txt'), 'w') 

	counts_per_hr = {}
	i = 0

	tweet = json.loads(f.readline())
	hour_start = tweet['firstpost_date']
	hour_end = hour_start + 3600
	counts_per_hr[hour_start] = 1

	while 1:
		i += 1
		line = f.readline()
		if line == '':
			break

		tweet = json.loads(line)
		tweet_time = tweet['firstpost_date']

		if tweet_time >= hour_start and tweet_time < hour_end:
			counts_per_hr[hour_start] += 1
		else:
			#print '[' + str(hour_start) + ' to ' + str(hour_end) + ')\t' + str(counts_per_hr[hour_start])
			out.write(str(hour_start) + '\t' + str(hour_end) + '\t' + str(counts_per_hr[hour_start]) + '\n' )
			
			hour_start = hour_end
			hour_end = hour_start + 3600
			counts_per_hr[hour_start] = 1

	#print '[' + str(hour_start) + ' to ' + str(hour_end) + ')\t' + str(counts_per_hr[hour_start])
	out.write(str(hour_start) + '\t' + str(hour_end) + '\t' + str(counts_per_hr[hour_start]) + '\n' )

	# print i
	# print counts_per_hr

	f.close()
	out.close()

if __name__ == "__main__":
	hashtags = ['#superbowl', '#nfl'];

	get_num_tweets_hr(hashtags[0])