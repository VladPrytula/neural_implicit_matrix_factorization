import math
import pandas as pd


class RecoMetrics():
    def __init__(self,
                 top_k=4
                 ):
        self._top_k = top_k
        self._ranked_users_scores_df = None
        # form positive interactions dataset

    def set_interactions(self, test_users, test_items, test_scores, 
                            negative_interactions_users, negative_items, negative_scores):
        positive_interactions_df = pd.DataFrame(
            {
                'users': test_users,
                'test_items': test_items,
                'test_scores': test_scores
            }
        )

        # and together with negative interactions
        eval_df = pd.DataFrame(
            {
                'users': negative_interactions_users + test_users,
                'items': negative_items + test_items,
                'scores': negative_scores + test_scores
            }
        )

        # we need this dumb join in order to be able to lately compute top hits
        eval_df = pd.merge(eval_df, positive_interactions_df,
                           on=['users'], how='left')

        eval_df['rank'] = eval_df.groupby('users')['scores'].rank(
            method='first', ascending=False)
        eval_df.sort_values(['users', 'rank'], inplace=True)
        self._ranked_users_scores_df = eval_df

    def cal_hit_ratio(self):
        top_k_df = self._ranked_users_scores_df[self._ranked_users_scores_df['rank'] <= self._top_k]
        test_in_top_k = top_k_df[top_k_df['test_items'] == top_k_df['items']]
        # TODO: the return statement shoudl be re-written to exlude multiple nuniqe calculations
        return len(test_in_top_k) * 1.0 / self._ranked_users_scores_df['users'].nunique()

    def cal_ndcg(self):
        raise NotImplementedError()

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k
