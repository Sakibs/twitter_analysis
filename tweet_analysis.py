import os
import json
import tarfile
import datetime, time
import numpy as np
import matplotlib.pyplot as plt


def TStoDT(tstamp):
	return datetime.datetime.fromtimestamp(
        tstamp
    ).strftime('%Y-%m-%d %H:%M:%S')

def get_TOD(tstamp):
	tstr = TStoDT(tstamp)
	elems = tstr.split(' ')
	clk = elems[1].split(':')
	return int(clk[0])

def store_stats_data(hashtag, hours, stats):
	out = open(os.path.join('.', 'part2', 'stats_'+hashtag+'.txt'), 'w')
	# for hr in hours:
	# 	out.write(TStoDT(hr)+'\n')
	# out.write('\n')
	for stat in stats:
		stat['DT_start'] = TStoDT(stat['hour_start'])
		out.write(json.dumps(stat)+'\n')
	out.close()

def get_tweet_stats(hashtag):
	filename = 'tweets_'+hashtag+'.txt'
	filepath = os.path.join('.', 'data', filename)
	f = open(filepath, 'r')

	out = open(os.path.join('.', 'part1', 'twts_hr_'+hashtag+'.txt'), 'w') 

	hour_list = []
	stats_list = []

	tweet = json.loads(f.readline())

	retweet_count = tweet['metrics']['citations']['total']
	user_followers = tweet['original_author']['followers']
	hour_start = tweet['firstpost_date']
	hour_end = hour_start + 3600

	current_stats = {
		'hour_start'		: hour_start,			# hour start and end window
		'hour_end'			: hour_end,				
		'n_tweets' 			: 1,					# count of # of tweets in hour span
		'n_retweets' 		: retweet_count,		# total # of retweets ** Verify
		'num_follwr'		: user_followers,		# total # of followers of users posting this hashtag
		'maxn_follwr'		: user_followers,		# max # followers in users ** NOT SURE YET
		'sum_follwr_post'	: 0, 					# sum of followers posting hashtag ** NOT SURE YET
		'tod'				: get_TOD(hour_start)
	}

	while 1:
		line = f.readline()
		if line == '':
			break

		tweet = json.loads(line)
		tweet_time = tweet['firstpost_date']
		retweet_count = tweet['metrics']['citations']['total']
		user_followers = tweet['original_author']['followers']

		if tweet_time >= hour_start and tweet_time < hour_end:
			# update current_stats
			current_stats['n_tweets'] += 1
			current_stats['n_retweets'] += retweet_count
			current_stats['num_follwr'] += user_followers
			current_stats['maxn_follwr'] = max(current_stats['maxn_follwr'], user_followers)
		else:
			# setup for next window
			# append hour to hours list. This is to keep track of order
			hour_list.append(hour_start)
			stats_list.append(current_stats)

			#print '[' + str(hour_start) + ' to ' + str(hour_end) + ')\t' + str(current_stats[hour_start])
			out.write(str(hour_start) + '\t' + str(hour_end) + '\t' + str(current_stats['n_tweets']) + '\n' )
			
			hour_start = hour_end
			hour_end = hour_start + 3600
			# reset current_stats
			current_stats = {
				'hour_start'		: hour_start,			# hour start and end window
				'hour_end'			: hour_end,				
				'n_tweets' 			: 1,					# count of number of tweets in hour span
				'n_retweets' 		: retweet_count,		# total number of retweets ** Verify
				'num_follwr'		: user_followers,		# total # of followers of users posting this hashtag
				'maxn_follwr'		: user_followers,		# max # followers in users ** NOT SURE YET
				'sum_follwr_post'	: 0, 					# sum of followers posting hashtag ** NOT SURE YET
				'tod'				: get_TOD(hour_start)
			}

	# append the last stats to the list
	hour_list.append(hour_start)
	stats_list.append(current_stats)

	out.write(str(hour_start) + '\t' + str(hour_end) + '\t' + str(current_stats['n_tweets']) + '\n' )

	store_stats_data(hashtag, hour_list, stats_list)

	f.close()
	out.close()

	return stats_list

def get_hist(hashtag):
	filename = 'twts_hr_'+hashtag+'.txt'
	filepath = os.path.join('.', 'part1', filename)
	f = open(filepath, 'r')

	lines = f.readlines()
	
	counts = []
	hours = []

	for line in lines:
		line = line[0:-1]
		items = line.split('\t')
		counts.append(int(items[2]))
		hours.append(items[0])

	hist, bins = np.histogram(counts, bins=len(counts))
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.bar(center, hist, align='center')
	plt.show()


if __name__ == "__main__":
	hashtags = ['#superbowl', '#nfl', '#gopatriots'];

	get_tweet_stats(hashtags[2])
	# get_hist(hashtags[0])