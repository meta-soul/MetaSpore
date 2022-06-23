import csv
import json
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--user-data',
        type=str,
        default='data/moviedata-10m/users.csv'
    )
    parser.add_argument(
        '--movie-data',
        type=str,
        default='data/moviedata-10m/movies.csv'
    )
    parser.add_argument(
        '--rating-data',
        type=str,
        default='data/moviedata-10m/ratings.csv'
    )
    parser.add_argument(
        '--movie-save',
        type=str,
        default='data/moviedata-10m-split/movie.json'
    )
    parser.add_argument(
        '--train-save',
        default='data/moviedata-10m-split/train.csv'
    )
    parser.add_argument(
        '--test-save',
        default='data/moviedata-10m-split/test.csv'
    )
    return parser.parse_args()

def load_csv_data(csv_file, delimiter=',', quotechar='"'):
    with open(csv_file, newline='', encoding='utf-8-sig') as fin:
        reader = csv.DictReader(fin, delimiter=delimiter, quotechar=quotechar)
        for row in reader:
            yield row

def load_users(args):
    users = []
    for user in load_csv_data(args.user_data):
        users.append({
            'user_id': user['USER_MD5'],
            'user_name': user['USER_NICKNAME']
        })
    return users

def load_movies(args):
    movies = []
    for movie in load_csv_data(args.movie_data):
        title = movie['NAME'].strip()
        genres = movie['GENRES'].strip().split('/')
        desc = movie['STORYLINE'].strip()
        tags = movie['TAGS'].strip().split('/')
        movies.append({
            'movie_id': movie['MOVIE_ID'],
            'title': title,
            'description': desc,
            'genres': [t.strip() for t in genres if t.strip()],
            'tags': [t.strip() for t in tags if t.strip()]
        })
    return movies

def load_ratings(args):
    ratings = []
    for rating in load_csv_data(args.rating_data):
        ratings.append({
            'user_id': rating['USER_MD5'],
            'movie_id': rating['MOVIE_ID'],
            'rating': rating['RATING'],
            'timestamp': int(time.mktime(time.strptime(rating['RATING_TIME'], '%Y-%m-%d %H:%M:%S')))
        })
    return ratings

def save_json(file, data):
    with open(file, 'w', encoding='utf8') as f:
        for row in data:
            print(json.dumps(row, ensure_ascii=False), file=f)

def main():
    args = parse_args()
    users = load_users(args)
    movies = load_movies(args)
    ratings = load_ratings(args)

    print(users[:5])
    print(movies[:5])
    print(ratings[:5])

    movie_map = {}
    for movie in movies:
        movie_map[movie['movie_id']] = movie
    save_json(args.movie_save, movies)

    user_actions = {}
    for rating in ratings:
        if rating['user_id'] not in user_actions:
            user_actions[rating['user_id']] = []
        user_actions[rating['user_id']].append(rating)
    
    min_num_actions = 6
    action_lens, valid_action_lens = [], []
    for user_id in user_actions:
        actions = user_actions[user_id]
        actions = sorted(actions, key=lambda x:x['timestamp'])
        user_actions[user_id] = actions
        action_lens.append(len(actions))
        if len(actions) >= min_num_actions:
            valid_action_lens.append(len(actions))
    
    print('user', len(action_lens))
    print('action min', min(action_lens))
    print('action max', max(action_lens))
    print('action avg', sum(action_lens)/len(action_lens))

    print('valid user', len(valid_action_lens))
    print('valid action min', min(valid_action_lens))
    print('valid action max', max(valid_action_lens))
    print('valid action avg', sum(valid_action_lens)/len(valid_action_lens))

    test_num_actions = 3
    train_data, test_data = [], []
    for user_id, actions in user_actions.items():
        if len(actions) < min_num_actions:
            continue 
        train_actions, test_actions = actions[:-test_num_actions], actions[-test_num_actions:]
        last_item_id = ''
        for rating in train_actions:
            train_data.append([
                1, rating['user_id'], '', '', '', '', rating['movie_id'], '', '', rating['rating'], last_item_id, '', rating['timestamp']
            ])
            last_item_id = rating['movie_id']
        last_item_id = ''
        for rating in test_actions:
            test_data.append([
                1, rating['user_id'], '', '', '', '', rating['movie_id'], '', '', rating['rating'], last_item_id, '', rating['timestamp']
            ])
            last_item_id = rating['movie_id']
    print('train size', len(train_data))
    print('test size', len(test_data))

    with open(args.train_save, 'w', encoding='utf8') as f:
        print("label,user_id,gender,age,occupation,zip,movie_id,recent_movie_ids,genre,rating,last_movie,last_genre,timestamp", file=f)
        for row in train_data:
            row = ','.join([str(i) for i in row])
            print(row, file=f)

    with open(args.test_save, 'w', encoding='utf8') as f:
        print("label,user_id,gender,age,occupation,zip,movie_id,recent_movie_ids,genre,rating,last_movie,last_genre,timestamp", file=f)
        for row in test_data:
            row = ','.join([str(i) for i in row])
            print(row, file=f)

if __name__ == '__main__':
    main()
