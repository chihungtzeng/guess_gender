# -*- coding: utf-8 -*-
"""
Constants about last names. (姓)
"""
import io
import os
import logging


LAST_NAMES = frozenset(list(
    u"趙錢孫李周吳鄭王馮陳褚衛蔣沈韓楊朱秦尤許何呂施張孔曹嚴華金魏陶姜戚謝"
    u"鄒喻柏水竇章雲蘇潘葛奚范彭郎魯韋昌馬苗鳳花方俞任袁柳酆鮑史唐費廉岑薛"
    u"雷賀倪湯滕殷羅畢郝鄔安常樂于時傅皮卞齊康伍余元卜顧孟平黃和穆蕭尹姚邵"
    u"湛汪祁毛禹狄米貝明臧計伏成戴談宋茅龐熊紀舒屈項祝董梁杜阮藍閔席季麻強"
    u"賈路婁危江童顏郭梅盛林刁鍾徐邱駱高夏蔡田樊胡凌霍虞萬支柯昝管盧莫經房"
    u"裘繆干解應宗丁宣賁鄧郁單杭洪包諸左石崔吉鈕龔程嵇邢滑裴陸榮翁荀羊於惠"
    u"甄麴家封芮羿儲靳汲邴糜松井段富巫烏焦巴弓牧隗山谷車侯宓蓬全郗班仰秋仲"
    u"伊宮甯仇欒暴甘鈄厲戎祖武符劉景詹束龍葉幸司韶郜黎薊薄印宿白懷蒲邰從鄂"
    u"索咸籍賴卓藺屠蒙池喬陰鬱胥能蒼雙聞莘黨翟譚貢勞逄姬申扶堵冉宰酈雍郤璩"
    u"桑桂濮牛壽通邊扈燕冀郟浦尚農溫別莊晏柴瞿閻充慕連茹習宦艾魚容向古易慎"
    u"戈廖庾終居衡步都耿滿弘匡國文寇廣祿闕東歐殳沃利蔚越夔隆師鞏厙聶晁勾"
    u"敖融冷訾辛闞那簡饒空曾毋沙乜養鞠須豐巢關蒯相查后荊紅游竺權逯蓋益桓"
    u"麥陽涂凃鐘官温粘衣辜岳黄"))

LAST_NAMES_2 = frozenset([
    u"司馬", u"上官", u"歐陽", u"夏侯", u"諸葛", u"東方", u"赫連", u"皇甫",
    u"尉遲", u"公羊", u"淳于", u"單于", u"太叔", u"申屠", u"公孫", u"仲孫",
    u"軒轅", u"令狐", u"鍾離", u"宇文", u"長孫", u"慕容", u"鮮于", u"閭丘",
    u"司徒", u"司空", u"司寇", u"呼延", u"左丘", u"東門", u"西門", u"張簡"])


def __get_all_names():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    names_file = os.path.join(cur_dir, "all_names.txt")
    with io.open(names_file, encoding="utf-8") as _fp:
        contents = _fp.read()
    return contents.splitlines()


def __rm_last_name(name):
    name_len = len(name)
    if name_len < 2:
        logging.warning(u"Unexpected name: %s", name)
        return ""
    if name_len == 2:
        if name[0] in LAST_NAMES:
            return name[1:]
    if name_len == 3:
        if name[0:2] in LAST_NAMES_2:
            return name[2:]
        elif name[0] in LAST_NAMES:
            return name[1:]
    if name_len == 4:
        if name[0:2] in LAST_NAMES_2:
            return name[2:]
        elif name[0] in LAST_NAMES and name[1] in LAST_NAMES:
            return name[2:]
    return name


def main():
    names = __get_all_names()
    for name in names:
        given_name = __rm_last_name(name)
        if given_name:
            print(given_name)

if __name__ == "__main__":
    main()
