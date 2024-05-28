import unittest
from ai_model.lda_model import LDAModel

dummy_dataset = [
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 정부의 민관합동 청년 일자리]",
    "[워싱턴=AP/뉴시스]12일(현지시간) 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
]

dummy_data_1 = "[ 문재인 대통령이 지난해 12월 27일 청와대에서 정부의 민관합동 청년 일자리]"
dummy_data_2 = "[워싱턴=AP/뉴시스]12일(현지시간) 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]"
dummy_data_3 = "[ 문재인 대통령이 지난해 12월 27일 청와대에서 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]"


class TestLdaModel(unittest.TestCase):

    # def test_all_process(self):
    #     ldaModel = LDAModel()
    #     ldaModel.train_lda_model(dataset=dummy_dataset)
    #     group_id, topic_distribution = ldaModel.get_group_id_and_topic_distribution(text=dummy_data)

    #     print("그룹 id 테스트", group_id)
    #     print("토픽 분포 테스트", topic_distribution)

    # def test_train_lda_model(self):
    #     LDAModel().train_lda_model(dataset=dummy_dataset)

    def test_lda_predict(self):
        group_id_1, topic_distribution_1 = LDAModel().get_group_id_and_topic_distribution(text=dummy_data_1)
        group_id_2, topic_distribution_2 = LDAModel().get_group_id_and_topic_distribution(text=dummy_data_2)
        group_id_3, topic_distribution_3 = LDAModel().get_group_id_and_topic_distribution(text=dummy_data_3)

        print("그룹 id 테스트", group_id_1)
        print("토픽 분포 테스트", topic_distribution_1)
        print()
        print("그룹 id 테스트", group_id_2)
        print("토픽 분포 테스트", topic_distribution_2)
        print()
        print("그룹 id 테스트", group_id_3)
        print("토픽 분포 테스트", topic_distribution_3)


if __name__ == "__main__":
    unittest.main()


# [['문재인', '대통령', '지난해', '청와대', '정부', '민관', '합동', '청년', '일자리'], ['워싱턴', '뉴시스', '현지', '미국', '워싱턴', '백악관', '루즈벨트', '룸', '열린', '반']]
