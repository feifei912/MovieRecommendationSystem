# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')


class MovieRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.user_movie_matrix = None
        self.model_knn = None

    def load_data(self):
        """加载电影数据集"""
        print("正在加载数据...")

        # 下载电影评分数据集
        try:
            # 尝试使用pandas读取在线数据集
            self.movies_df = pd.read_csv('https://raw.githubusercontent.com/rounakbanik/MovieLens/master/movies.csv')
            self.ratings_df = pd.read_csv('https://raw.githubusercontent.com/rounakbanik/MovieLens/master/ratings.csv')
        except:
            # 如果在线加载失败，使用生成的示例数据
            print("无法加载在线数据，使用示例数据...")
            # 生成示例电影数据
            movie_ids = list(range(1, 101))
            movie_titles = [f"Movie {i}" for i in movie_ids]
            genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Sci-Fi', 'Thriller', 'Romance', 'Horror']
            movie_genres = ["|".join(np.random.choice(genres, size=np.random.randint(1, 4), replace=False)) for _ in
                            range(100)]

            self.movies_df = pd.DataFrame({
                'movieId': movie_ids,
                'title': movie_titles,
                'genres': movie_genres
            })

            # 生成示例评分数据
            user_ids = np.repeat(range(1, 101), 20)  # 100个用户，每人评价20部电影
            movie_ids = np.random.choice(range(1, 101), size=100 * 20)  # 随机选择电影
            ratings = np.random.uniform(1, 5, size=100 * 20).round(1)  # 随机评分1-5
            timestamps = np.random.randint(1000000000, 1600000000, size=100 * 20)  # 随机时间戳

            self.ratings_df = pd.DataFrame({
                'userId': user_ids,
                'movieId': movie_ids,
                'rating': ratings,
                'timestamp': timestamps
            })

        print(
            f"加载完成! 数据集包含 {self.movies_df.shape[0]} 部电影和 {self.ratings_df['userId'].nunique()} 个用户的 {self.ratings_df.shape[0]} 条评分。")

    def analyze_data(self):
        """分析数据，显示基本信息"""
        print("\n===== 数据分析 =====")

        # 电影数据基本信息
        print("\n电影数据集信息:")
        print(f"总电影数: {self.movies_df.shape[0]}")

        # 显示前5部电影
        print("\n前5部电影示例:")
        print(self.movies_df.head())

        # 评分数据基本信息
        print("\n评分数据集信息:")
        print(f"总评分数: {self.ratings_df.shape[0]}")
        print(f"用户数: {self.ratings_df['userId'].nunique()}")

        # 评分分布
        plt.figure(figsize=(10, 6))
        sns.histplot(self.ratings_df['rating'], bins=10, kde=True)
        plt.title('评分分布')
        plt.xlabel('评分')
        plt.ylabel('频次')
        plt.savefig('rating_distribution.png')
        print("评分分布图已保存为 'rating_distribution.png'")

    def preprocess_data(self):
        """预处理数据"""
        print("\n===== 数据预处理 =====")

        # 合并电影数据和评分数据
        df = pd.merge(self.ratings_df, self.movies_df, on='movieId')

        # 统计每部电影的评分数
        movie_count = df.groupby('title')['rating'].count().reset_index(name='count')

        # 只保留评分数超过10的电影
        popular_movies = movie_count[movie_count['count'] >= 10]
        df_popular = df.merge(popular_movies, on='title')

        print(f"过滤后保留了 {len(popular_movies)} 部评分数超过10的电影")

        # 创建用户-电影评分矩阵
        user_movie_df = df_popular.pivot_table(index='userId', columns='title', values='rating')

        # 填充NaN值
        self.user_movie_matrix = user_movie_df.fillna(0)

        print(f"创建了大小为 {self.user_movie_matrix.shape} 的用户-电影矩阵")

    def build_model(self):
        """构建基于物品的协同过滤推荐模型"""
        print("\n===== 构建推荐模型 =====")

        # 计算电影之间的相似度
        movie_similarity = cosine_similarity(self.user_movie_matrix.T)
        movie_similarity_df = pd.DataFrame(movie_similarity,
                                           index=self.user_movie_matrix.columns,
                                           columns=self.user_movie_matrix.columns)

        # 创建稀疏矩阵加速KNN搜索
        ratings_matrix_csr = csr_matrix(self.user_movie_matrix.values)

        # 训练KNN模型
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model_knn.fit(ratings_matrix_csr.T)  # 转置以获得电影之间的相似度

        print("成功构建KNN推荐模型")

    def recommend_movies(self, movie_name, n_recommendations=10):
        """根据指定电影推荐相似电影"""
        print(f"\n===== 基于《{movie_name}》的电影推荐 =====")

        # 检查电影是否在数据集中
        if movie_name not in self.user_movie_matrix.columns:
            most_similar = self.movies_df[self.movies_df['title'].str.contains(movie_name, case=False)]
            if not most_similar.empty:
                movie_name = most_similar.iloc[0]['title']
                print(f"未找到完全匹配的电影，使用最相似的电影: {movie_name}")
            else:
                print(f"未找到电影 '{movie_name}'，请尝试其他电影名")
                # 显示随机5部电影作为建议
                random_movies = self.user_movie_matrix.columns.tolist()
                np.random.shuffle(random_movies)
                print("您可以尝试这些电影:", random_movies[:5])
                return

        # 获取电影的索引
        movie_idx = self.user_movie_matrix.columns.get_loc(movie_name)

        # 使用KNN查找最相似的电影
        distances, indices = self.model_knn.kneighbors(
            self.user_movie_matrix.iloc[:, movie_idx].values.reshape(1, -1).T,
            n_neighbors=n_recommendations + 1
        )

        # 获取推荐电影列表
        movie_indices = indices.flatten()[1:]  # 排除电影本身
        recommended_movies = self.user_movie_matrix.columns[movie_indices].tolist()

        # 显示推荐结果
        print(f"\n为您推荐的与《{movie_name}》相似的 {n_recommendations} 部电影:")
        for i, movie in enumerate(recommended_movies, 1):
            print(f"{i}. {movie}")

        return recommended_movies

    def recommend_for_user(self, user_id, n_recommendations=10):
        """为指定用户推荐电影"""
        print(f"\n===== 为用户 {user_id} 推荐电影 =====")

        # 检查用户是否在数据集中
        if user_id not in self.user_movie_matrix.index:
            print(f"用户 {user_id} 不在数据集中")
            return

        # 获取用户已评价和未评价的电影
        user_ratings = self.user_movie_matrix.loc[user_id]
        watched_movies = user_ratings[user_ratings > 0].index.tolist()
        unwatched_movies = user_ratings[user_ratings == 0].index.tolist()

        print(f"用户已评价 {len(watched_movies)} 部电影，未评价 {len(unwatched_movies)} 部电影")

        if not watched_movies:
            print("用户尚未评价任何电影，无法提供个性化推荐")
            return

        # 为每部未观看的电影预测评分
        predicted_ratings = {}
        for movie in unwatched_movies:
            # 计算与该电影的相似度
            movie_similarity = cosine_similarity(
                self.user_movie_matrix[watched_movies].T,
                self.user_movie_matrix[[movie]].T
            ).flatten()

            # 计算加权平均评分
            user_ratings_for_watched = user_ratings[watched_movies].values
            predicted_rating = np.sum(movie_similarity * user_ratings_for_watched) / np.sum(np.abs(movie_similarity))
            predicted_ratings[movie] = predicted_rating

        # 排序并获取推荐电影
        recommended_movies = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

        # 显示推荐结果
        print(f"\n为用户 {user_id} 推荐的 {n_recommendations} 部电影:")
        for i, (movie, rating) in enumerate(recommended_movies, 1):
            print(f"{i}. {movie} (预测评分: {rating:.2f})")

        return [movie for movie, _ in recommended_movies]

    def evaluate_model(self):
        """评估推荐模型性能"""
        print("\n===== 模型评估 =====")

        # 准备训练集和测试集
        # 只使用非零评分
        ratings_matrix = self.user_movie_matrix.values
        non_zeros = ratings_matrix.nonzero()
        ratings_array = ratings_matrix[non_zeros]

        # 划分训练集和测试集
        train_indices, test_indices = train_test_split(
            np.arange(len(non_zeros[0])),
            test_size=0.2
        )

        # 创建训练集矩阵
        train_matrix = ratings_matrix.copy()
        user_indices = non_zeros[0][test_indices]
        item_indices = non_zeros[1][test_indices]
        train_matrix[user_indices, item_indices] = 0

        # 预测测试集评分
        predictions = []
        actuals = ratings_array[test_indices]

        for user_idx, item_idx in zip(user_indices, item_indices):
            user_ratings = train_matrix[user_idx]
            rated_items = user_ratings.nonzero()[0]

            if len(rated_items) == 0:
                # 如果用户没有评价任何电影，使用平均评分
                prediction = np.mean(ratings_array)
            else:
                # 计算该电影与用户已评价电影的相似度
                item_similarities = cosine_similarity(
                    train_matrix[:, rated_items].T,
                    train_matrix[:, item_idx].reshape(1, -1)
                ).flatten()

                # 计算加权平均评分
                prediction = np.sum(item_similarities * user_ratings[rated_items]) / np.sum(np.abs(item_similarities))

            predictions.append(prediction)

        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = np.mean(np.abs(np.array(predictions) - actuals))

        print(f"基于协同过滤的推荐模型性能:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")


def main():
    # 创建推荐系统
    recommender = MovieRecommendationSystem()

    # 加载数据
    recommender.load_data()

    # 数据分析
    recommender.analyze_data()

    # 数据预处理
    recommender.preprocess_data()

    # 构建模型
    recommender.build_model()

    # 评估模型
    recommender.evaluate_model()

    # 基于电影推荐
    recommender.recommend_movies("Toy Story", n_recommendations=5)

    # 基于用户推荐
    recommender.recommend_for_user(1, n_recommendations=5)

    print("\n===== 交互式推荐 =====")
    while True:
        print("\n请选择操作:")
        print("1. 基于电影的推荐")
        print("2. 基于用户的推荐")
        print("3. 退出")

        choice = input("请输入选择 (1-3): ")

        if choice == "1":
            movie_name = input("请输入电影名称: ")
            n_recommendations = int(input("推荐数量: "))
            recommender.recommend_movies(movie_name, n_recommendations)

        elif choice == "2":
            try:
                user_id = int(input("请输入用户ID: "))
                n_recommendations = int(input("推荐数量: "))
                recommender.recommend_for_user(user_id, n_recommendations)
            except ValueError:
                print("用户ID应为整数")

        elif choice == "3":
            print("感谢使用推荐系统!")
            break

        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()
