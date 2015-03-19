import os
import json
import tarfile
import datetime, time
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from math import sqrt


def TStoDT(tstamp):
	return datetime.datetime.fromtimestamp(
        tstamp
    ).strftime('%Y-%m-%d %H:%M:%S')

def get_TOD(tstamp):
	tstr = TStoDT(tstamp)
	elems = tstr.split(' ')
	clk = elems[1].split(':')
	return int(clk[0])

def getHourStart(tstamp):
	tstr = TStoDT(tstamp)
	elems = tstr.split(' ')
	clk = elems[1].split(':')
	seconds = int(clk[1])*60+int(clk[2])

	return tstamp-seconds

def store_stats_data(hashtag, hours, stats):
	out = open(os.path.join('.', 'part2', 'stats_'+hashtag+'.txt'), 'w')
	# for hr in hours:
	# 	out.write(TStoDT(hr)+'\n')
	# out.write('\n')
	for stat in stats:
		stat['DT_start'] = TStoDT(stat['hour_start'])
		out.write(json.dumps(stat)+'\n')
	out.close()

def load_stats_data(hashtag):
	f = open(os.path.join('.', 'part2', 'stats_'+hashtag+'.txt'), 'r')
	#f = open(os.path.join('.', 'part2', hashtag +'.txt'), 'r')

	stats_list = []
	for line in f:
		stats_list.append(json.loads(line))

	return stats_list

def load_split_stats_data(hashtag, periodnum):
	f = open(os.path.join('.', 'part4', 'stats_'+hashtag+ '_' + periodnum + '.txt'), 'r')

	stats_list = []
	for line in f:
		stats_list.append(json.loads(line))

	return stats_list

def split_stats_data(hashtag, times):
	stats = load_stats_data(hashtag)

	if len(times) <= 0:
		return

	out = open(os.path.join('.', 'part4', 'stats_'+hashtag+'_period1'+'.txt'), 'w')

	i=0
	t=0
	while i < len(stats):
		if(t < len(times) and stats[i]['hour_start'] >= times[t]):
			# print "11111"
			out.close()
			out = open(os.path.join('.', 'part4', 'stats_'+hashtag+'_'+'period'+str(t+2)+'.txt'), 'w')
			t += 1
		else:
			# print "22222"
			out.write(json.dumps(stats[i])+'\n')
			i += 1
	out.close()
			
def update_stat_item(tweet, stat):
	tweet_time = tweet['firstpost_date']
	retweet_count = tweet['metrics']['citations']['total']
	user_followers = tweet['original_author']['followers']

	stat['n_tweets'] += 1
	stat['n_retweets'] += retweet_count
	stat['num_follwr'] += user_followers
	stat['maxn_follwr'] = max(stat['maxn_follwr'], user_followers)

def get_tweet_stats(hashtag):
	#filename = 'tweets_'+hashtag+'.txt'
	filename = hashtag + '.txt'  #HACK SOLUTION HERE
	#filepath = os.path.join('.', 'data', filename)
	filepath = os.path.join('.', 'test_data', filename) #HACK SOLUTION HERE
	f = open(filepath, 'r')

	out = open(os.path.join('.', 'part1', 'twts_hr_'+hashtag+'.txt'), 'w') 

	hour_list = []
	stats_list = []

	tweet = json.loads(f.readline())

	retweet_count = tweet['metrics']['citations']['total']
	hour_start = getHourStart( tweet['firstpost_date'] )
	hour_end = hour_start + 3600

	current_stats = {
		'hour_start'		: hour_start,			# hour start and end window
		'hour_end'			: hour_end,				
		'n_tweets' 			: 0,					# count of # of tweets in hour span
		'n_retweets' 		: 0,		# total # of retweets ** Verify
		'num_follwr'		: 0,		# total # of followers of users posting this hashtag
		'maxn_follwr'		: 0,		# max # followers in users ** NOT SURE YET
		'sum_follwr_post'	: 0, 					# sum of followers posting hashtag ** NOT SURE YET
		'tod'				: get_TOD(hour_start)
	}

	while 1:
		tweet_time = tweet['firstpost_date']
	
		if tweet_time >= hour_start and tweet_time < hour_end:
			# update current_stats
			update_stat_item(tweet, current_stats)
		else:
			# setup for next window
			# append hour to hours list. This is to keep track of order
			hour_list.append(hour_start)
			stats_list.append(current_stats)

			#print '[' + str(hour_start) + ' to ' + str(hour_end) + ')\t' + str(current_stats[hour_start])
			out.write(str(hour_start) + '\t' + str(hour_end) + '\t' + str(current_stats['n_tweets']) + '\n' )
			
			found_interval = False
			while not found_interval:
				hour_start = hour_end
				hour_end = hour_start + 3600

				# reset current_stats
				current_stats = {
					'hour_start'		: hour_start,			# hour start and end window
					'hour_end'			: hour_end,				
					'n_tweets' 			: 0,					# count of number of tweets in hour span
					'n_retweets' 		: 0,		# total number of retweets ** Verify
					'num_follwr'		: 0,		# total # of followers of users posting this hashtag
					'maxn_follwr'		: 0,		# max # followers in users ** NOT SURE YET
					'sum_follwr_post'	: 0, 					# sum of followers posting hashtag ** NOT SURE YET
					'tod'				: get_TOD(hour_start)
				}

				if tweet_time >= hour_start and tweet_time < hour_end:
					found_interval = True
				else:
					hour_list.append(hour_start)
					stats_list.append(current_stats)
					out.write(str(hour_start) + '\t' + str(hour_end) + '\t' + str(current_stats['n_tweets']) + '\n' )

			update_stat_item(tweet, current_stats)

		line = f.readline()
		if line == '':
			break

		tweet = json.loads(line)

	# append the last stats to the list
	hour_list.append(hour_start)
	stats_list.append(current_stats)

	out.write(str(hour_start) + '\t' + str(hour_end) + '\t' + str(current_stats['n_tweets']) + '\n' )

	store_stats_data(hashtag, hour_list, stats_list)

	f.close()
	out.close()

def plot_hist(hashtag):
	tweet_stats = get_tweet_stats(hashtag)
	x_arr = np.ones((len(tweet_stats),1))
	n_tweets = get_feature_array(tweet_stats,'n_tweets',x_arr)
	print n_tweets

	plt.hist(n_tweets, bins = 1000)
	plt.ylim(0, 50)
	plt.show()

def make_input_matrix(tweet_stats,time_idx):
	n_tweets = get_feature_array(tweet_stats,'n_tweets',time_idx)
 	n_retweets = get_feature_array(tweet_stats,'n_retweets',time_idx)
	num_follwr = get_feature_array(tweet_stats, 'num_follwr',time_idx)
	maxn_follwr = get_feature_array(tweet_stats, 'maxn_follwr',time_idx)
	tod = get_feature_array(tweet_stats, 'tod', time_idx)

	#x = np.column_stack(( n_tweets, n_retweets, num_follwr, maxn_follwr, tod))
	x = np.column_stack(( np.ones((len(time_idx)-1,1)), n_tweets, n_retweets, num_follwr, maxn_follwr, tod))
	return x

def get_regression_model(tweet_stats,time_idx):
	y = get_output(tweet_stats, 'n_tweets', time_idx)
	x = make_input_matrix(tweet_stats,time_idx)
	
	model = sm.OLS(y,x)
	return model

def get_output(tweet_stats,feature,time_idx):

	y = np.zeros((len(time_idx)-1,1))
	max_idx = len(time_idx)-1
	for point in time_idx[:-1]:
		if(point-time_idx[0]>=max_idx): #THIS IS AN ERROR. SHOULD BE FIXED
			break
		y[point-time_idx[0],0] = tweet_stats[point+1][feature]

	return y

def get_feature_array(tweet_stats,feature,time_idx):
	x = np.zeros((len(time_idx)-1,1))
	max_idx = len(time_idx)-1
	for point in time_idx[:-1]:
		if(point-time_idx[0]>=max_idx): #THIS IS AN ERROR, SHOULD BE FIXED
			break
		x[point-time_idx[0],0] = tweet_stats[point][feature]

	return x

def cross_validate(tweet_stats):
	tot_lenth = len(tweet_stats)
	kf = cross_validation.KFold(n=tot_lenth,n_folds=10, shuffle = True)
	rms_error_arr = []
	for train_idx, test_idx in kf:
		x_arr  = make_input_matrix(tweet_stats,test_idx)
		
		model = get_regression_model(tweet_stats,train_idx)
		res = model.fit()
		newy =  res.predict(x_arr)
		
		actualy = get_feature_array(tweet_stats,'n_tweets',test_idx)
		rms_error = sqrt(mean_squared_error(actualy,newy))
		print ('rmse is ' + str(rms_error))
		rms_error_arr.append(rms_error)
	
	avg_error = float(sum(rms_error_arr))/len(rms_error_arr)
	print ('average error is ' + str(avg_error))

def plot_scatter(tweet_stats,feature,model):
	time_idx = np.arange(0,len(tweet_stats)-1)
	x = make_input_matrix(tweet_stats,time_idx)
	res = model.fit()
	ypred = res.predict(x)
	print ypred.shape

	feat_arr = get_feature_array(tweet_stats,feature,time_idx)
	print feat_arr.shape

	# Plotting predictant vs feature (predictant is x and feature is y)
	plt.scatter(ypred,feat_arr)
	plt.show()

def predict_next_hour(samplenum,periodnum):
	tweet_stats = load_split_stats_data(hashtags[0], periodnum)
	time_idx = np.arange(0,len(tweet_stats)-1)
	model = get_regression_model(tweet_stats,time_idx)
	results = model.fit()
	#print results.summary()

	get_tweet_stats(samplenum+'_'+periodnum)
	tweet_stats = load_stats_data(samplenum+'_'+periodnum)
	test_idx = np.arange(0,len(tweet_stats)-1)
	x_arr  = make_input_matrix(tweet_stats,test_idx)
	print x_arr.shape

	ypred = results.predict(x_arr)
	print ypred
	return ypred[-1]



if __name__ == "__main__":
	hashtags = ['#superbowl', '#nfl', '#gopatriots', '#gohawks', '#patriots', '#sb49'];

	#get_tweet_stats(hashtags[0])
	#tweet_stats = load_stats_data(hashtags[0])
	#cross_validate(tweet_stats)

	#time_idx = np.arange(0,len(tweet_stats)-1)
	#model = get_regression_model(tweet_stats,time_idx)
	#results = model.fit()
	#print results.summary()

	#results = model.fit()
	#print (results.summary())
	#plot_hist(hashtags[1])

	#plot_hist(hashtags[0])
	#plot_scatter(tweet_stats,'n_tweets', model)

	#split_stats_data(hashtags[0], [1422720000, 1422763200])
	print predict_next_hour('sample1','period1')
