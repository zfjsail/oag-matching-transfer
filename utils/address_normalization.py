# -*- coding:utf-8 -*-
import re


class addressNormalization():
    def trim_rfc822(self, s):
        first = "<rfc822>"
        last = "</rfc822>"
        try:
            start = s.index(first)
            end = s.index(last, start) + len(last)
            return s[:start] + s[end:]
        except ValueError:
            return s

    def trim_i(self, s):
        first = "<i>"
        last = "</i>"
        try:
            start = s.index(first)+len(first)
            end = s.index(last, start)
            return s[start:end]
        except ValueError:
            return s

    def trim_url(self, s):
        try:
            start = s.index('url:')
            return s[0:start]
        except:
            try:
                start2 = s.index('http:')
                return s[0:start2]
            except:
                return s

    def trimEmail(self, dest):
        raw = dest
        dest = dest.replace('&', ' ')
        dest = self.trim_i(dest)
        dest = self.trim_rfc822(dest)
        dest = self.trim_url(dest)
        try:
            start = dest.index('email:')
            return dest[0:start].strip()
        except:
            import re
            restring1 = r'\b[\w.-]+?@\w+?\.\w+?\b.\w+?\b'
            restring2 = r'\b[\w.-]+?@\w+?\.\w+?\b'
            # dest = u'computer graphics laboratory eth zürich switzerland grossm@inf.ethz.cn'.encode('utf8')
            # print dest
            newdest = re.sub(restring1, "", dest)
            newdest = re.sub(restring2, "", newdest)
            # m = re.search(restring,dest)
            # print m.group(0)
            if newdest == '':
                # domain1 = r'(?<=[@.][^.]+.[^.]+(?=\^))'
                # domain = r'(?<=@)[^.]+(?=\.)'
                start = raw.find('@')
                if start != -1:
                    strs = raw[start+1:].split('.')
                    if len(strs) >= 2:
                        if strs[-2] != 'gmail':
                            return strs[-2].strip()
            return newdest.strip()

    def findmianpart(self, dest):
        p = re.compile(r'\,|\.|\(|\)|\;')
        place = p.split(dest)
        flag = True
        for u in place:
            u = u.strip()
            p = r'(.*) ((UNIVERSITY)|(大学)|(研究院)|(中科院))(.*)'
            m = re.search(p, u)
            if m is not None:
                dest = m.group().strip()
                flag = False
        if flag:  # 再次进行处理
            dest = self.find_inst(dest)
        return dest

    def regex(self):
        ex_university = r"[\s\w]+OF [\s\w]?(UNIVERSITY OF[\s\w]*[\w]+)|"
        ex_university += r"[\s\w]+OF ([\s\w]+?UNIVERSITY)|"
        ex_university += r" ?([\w]+[\s\w]*UNIVERSITY(?: OF[\s\w]*[\w]+)?)|"
        ex_university += r"( ?UNIVERSITY OF[\s\w]+)|"
        ex_university += r"([\s\w]*UNIVERSITA[\s\w]*)|"
        ex_university += r"([\s\w]*UNIVERSIDAD[\s\w]*)|"
        ex_university += r"([\s\w]*UNIVERSITÉ[\s\w]*)|"
        ex_university += r"([\s\w]*UNIVERSIDAD[\s\w]*)|"
        ex_university += r"([\s\w]*UNIVERSITÄ[\s\w]*)|"
        ex_university += r"([\s\w]*UNIVERSITÀ[\s\w]*)"
        ex_univ = r"([\s\w\.]+ UNIV)|"
        ex_univ += r"(UNIV\.[\s\w]+)"
        ex_school = r"[\s\w]?OF [\s\w]?(SCHOOL OF[\s\w]*[\w]+)|"
        ex_school += r"[\s\w]+OF ([\s\w]+?SCHOOL)|"
        ex_school += r" ?([\w]+[\s\w]*SCHOOL(?: OF[\s\w]*[\w]+)?)"
        ex_college = r"[\s\w]?OF [\s\w]?(COLLEGE OF[\s\w]*[\w]+)|"
        ex_college += r"[\s\w]+OF ([\s\w]+?COLLEGE)|"
        ex_college += r" ?([\w]+[\s\w]*COLLEGE(?: OF[\s\w]*[\w]+)?)|"
        ex_college += r"(COLLEGE OF[\s\w]+)"
        ex_institute = r"[\s\w]?OF [\s\w]?(INSTITUTE (?:OF|FOR)[\s\w]*[\w]+)|"
        ex_institute += r"[\s\w]+OF ([\s\w]+?INSTITUTE)|"
        ex_institute += r"([\w]+[\s\w]*INSTITUTE(?: (?:OF|FOR)[\s\w]*[\w]+)?)|"
        ex_institute += r"(INSTITUTE (?:OF|FOR)[\s\w]+)"
        ex_inst = r"[\s\w]?OF [\s\w]?(INST\. (?:OF|FOR)[\s\w]*[\w]+)|"
        ex_inst += r"[\s\w]+OF ([\s\w]+?INST\.)|"
        ex_inst += r"([\w]+[\s\w\.]*INST\.(?:(?:OF|FOR)[\s\w]*[\w]+)?)|"
        ex_inst += r"(INST\. (?:OF|FOR)[\s\w]+)"
        ex_laboratory = r"[\s\w]?OF [\s\w]?(LABORATORY (?:OF|FOR)[\s\w]*[\w]+)|"
        ex_laboratory += r"[\s\w]+OF ([\s\w]+?LABORATORY)|"
        ex_laboratory += r"([\w]+[\s\w]*LABORATORY(?: (?:OF|FOR)[\s\w]*[\w]+)?)|"
        ex_laboratory += r"(LABORATORY (?:OF|FOR)[\s\w]+)|"
        ex_laboratory += r"([\s\w\.]+LAB)"
        ex_scenter = r"[\s\w]?OF [\s\w]?(RESEARCH CENTER (?:OF|FOR)[\s\w]*[\w]+)|"
        ex_scenter += r"[\s\w]+OF ([\s\w]+?RESEARCH CENTER)|"
        ex_scenter += r"([\w]+[\s\w]*RESEARCH CENTER(?: (?:OF|FOR)[\s\w]*[\w]+)?)|"
        ex_scenter += r"(RESEARCH CENTER (?:OF|FOR)[\s\w]+)"
        ex_center = r"[\s\w]?OF [\s\w]?(CENTER (?:OF|FOR)[\s\w]*[\w]+)|"
        ex_center += r"[\s\w]+OF ([\s\w]+?CENTER)|"
        ex_center += r"([\w]+[\s\w]*CENTER(?: (?:OF|FOR)[\s\w]*[\w]+)?)|"
        ex_center += r"(CENTER (?:OF|FOR)[\s\w]+)"
        ex_hospital = r"[\s\w]?OF [\s\w]?(HOSPITAL (?:OF|FOR)[\s\w]*[\w]+)|"
        ex_hospital += r"[\s\w]+OF ([\s\w]+?HOSPITAL)|"
        ex_hospital += r"([\w]+[\s\w]*HOSPITAL(?: (?:OF|FOR)[\s\w]*[\w]+)?)|"
        ex_hospital += r"(HOSPITAL (?:OF|FOR)[\s\w]+)|"
        ex_hospital += r" ?(HOSPITAL [\s\w]+)|([\s\w]*HÔPITAL[\s\w]*)"
        ex_corporation = r"[\s\w]?OF [\s\w]?(CORPORATION (?:OF|FOR)[\s\w]*[\w]+)|"
        ex_corporation += r"[\s\w]+OF ([\s\w]+?CORPORATION)|"
        ex_corporation += r"([\w]+[\s\w]*CORPORATION(?: (?:OF|FOR)[\s\w]*[\w]+)?)|"
        ex_corporation += r"(CORPORATION (?:OF|FOR)[\s\w]+)|"
        ex_corporation += r"[\s\w]+COMPANY(?: LIMITED)?|"
        ex_corporation += r"([\s\w]+CO\.)"
        ex_organization = r"[\s\w]?OF [\s\w]?(ORGANIZATION (?:OF|FOR)[\s\w]*[\w]+)|"
        ex_organization += r"[\s\w]+OF ([\s\w]+?ORGANIZATION)|"
        ex_organization += r"([\w]+[\s\w]*ORGANIZATION(?: (?:OF|FOR)[\s\w]*[\w]+)?)|"
        ex_organization += r"(ORGANIZATION (?:OF|FOR)[\s\w]+)"
        ex_bureau = r"[\s\w]?OF [\s\w]?(BUREAU OF[\s\w]*[\w]+)|"
        ex_bureau += r"[\s\w]+OF ([\s\w]+?BUREAU)|"
        ex_bureau += r"([\w]+[\s\w]*BUREAU(?: OF[\s\w]*[\w]+)?)|"
        ex_bureau += r"(BUREAU OF[\s\w]+)"
        ex_academy = r"[\s\w]?OF [\s\w]?(ACADEMY OF[\s\w]*[\w]+)|"
        ex_academy += r"[\s\w]+OF ([\s\w]+?ACADEMY)|"
        ex_academy += r"([\w]+[\s\w]*ACADEMY(?: OF[\s\w]*[\w]+)?)|"
        ex_academy += r"(ACADEMY OF[\s\w]+)"
        ex_university_chinese = r"([\u4e00-\u9fa5\s]+大学)"
        ex_hospital_chinese = r"([\u4e00-\u9fa5\s]+医院)"
        ex_school_chinese = r"([\u4e00-\u9fa5\s]+学院)|"
        ex_school_chinese += r"([\u4e00-\u9fa5\s]+学校)|"
        ex_school_chinese += r"([\u4e00-\u9fa5\s]+中学)|"
        ex_school_chinese += r"([\u4e00-\u9fa5\s]+党校)|"
        ex_school_chinese += r"([\u4e00-\u9fa5\s]+小学)|"
        ex_school_chinese += r"([\u4e00-\u9fa5\s]+中学)"
        ex_group_chinese = r"([\u4e00-\u9fa5\s]+集团)"
        ex_corporation_chinese = r"([\u4e00-\u9fa5\s]+公司)|"
        ex_gov_chinese = r"([\u4e00-\u9fa5\s]+电视台)|"
        ex_gov_chinese += r"([\u4e00-\u9fa5\s]+电台)|"
        ex_gov_chinese += r"([\u4e00-\u9fa5\s]+处)|"
        ex_gov_chinese += r"([\u4e00-\u9fa5\s]+站)|"
        ex_gov_chinese += r"([\u4e00-\u9fa5\s]+馆)|"
        ex_gov_chinese += r"([\u4e00-\u9fa5\s]+办公室)|"
        ex_gov_chinese += r"([\u4e00-\u9fa5\s]+办)|"
        ex_gov_chinese += r"([\u4e00-\u9fa5\s]+室)|"
        ex_gov_chinese += r"([\u4e00-\u9fa5\s]+银行)"
        ex_factory_chinese = r"([\u4e00-\u9fa5\s]+厂)"
        ex_academy_chinese = r"([0-9\u4e00-\u9fa5\s]+院)"
        ex_scenter_chinese = r"([0-9\u4e00-\u9fa5\s]+研究中心)"
        ex_institute_chinese = r"([0-9\u4e00-\u9fa5\s]+所)"
        ex_laboratory_chinese = r"([0-9\u4e00-\u9fa5\s]+实验室)"
        ex_newspaper_chinese = r"([\u4e00-\u9fa5\s]+社)|"
        ex_newspaper_chinese += r"([\u4e00-\u9fa5\s]+报)"
        ex_bureau_chinese = r"([\u4e00-\u9fa5\s]+局)"
        ex_center_chinese = r"([\u4e00-\u9fa5\s]+中心)"

        ex_institution_chinese = r"([\u4e00-\u9fa5\s]+厅)|"
        ex_institution_chinese += r"([\u4e00-\u9fa5\s]+会)|"
        ex_institution_chinese += r"([\u4e00-\u9fa5\s]+政府)|"
        ex_institution_chinese += r"([\u4e00-\u9fa5\s]+厅)|"
        ex_institution_chinese += r"([0-9\u4e00-\u9fa5\s]+部队)|"
        ex_institution_chinese += r"([\u4e00-\u9fa5\s]+委)|"
        ex_institution_chinese += r"([\u4e00-\u9fa5\s]+办公室)"
        ex_search_chinese = r"([0-9\u4e00-\u9fa5\s]+研究室)"
        ex_clinic_chinese = r"([0-9\u4e00-\u9fa5\s]+诊所)|"
        ex_clinic_chinese += r"([0-9\u4e00-\u9fa5\s]+门诊部)"
        ex_ministry_chinese = r"([\u4e00-\u9fa5\s]+部)"
        ex_observatory = r"([\u4e00-\u9fa5\s]+天文台)"
        ex_dang = r"([\u4e00-\u9fa5\s]+党委)"
        ex_da_group = r"([0-9\u4e00-\u9fa5\s]+大队)"
        ex_kexie = r"([0-9\u4e00-\u9fa5\s]+科协)"
        ex_service = r"([\s\w]*SERVICE[\s\w]*)"
        ex_ministry = r"([\s\w]*MINISTRY[\s\w]*)"
        ex_foundation = r"([\s\w]*FOUNDATION[\s\w]*)"
        ex_council = r"([\s\w]*COUNCIL[\s\w]*)"
        ex_fund = r"([\s\w]*FUND[\s\w]*)"
        ex_project = r"([\s\w]*PROJECT)"
        ex_coll = r"([\s\w\.]*COLL\.[\s\w\.]*)"
        ex_factory = r"([\s\w]*FACTORY[\s\w]*)"
        ex_agency = r"([\s\w]*AGENCY[\s\w]*)"
        ex_committee = r"([\s\w]*COMMITTEE[\s\w]*)"
        ex_commission = r"([\s\w]*COMMISSION[\s\w]*)"
        ex_group = r"([\s\w]*GROUP[\s\w]*)|"
        ex_group += r"[\s\w]?OF [\s\w]?(GROUP OF[\s\w]*[\w]+)|"
        ex_group += r"[\s\w]+OF ([\s\w]+?GROUP)|"
        ex_group += r"([\w]+[\s\w]*GROUP(?: OF[\s\w]*[\w]+)?)|"
        ex_group += r"(GROUP OF[\s\w]+)"
        ex_institut = r"([\s\w]*INSTITUTO?[\s\w]*)"
        ex_laboratoire = r"([\s\w]*LABORATOIRE[\s\w])|"
        ex_laboratoire += r"([\s\w]*LABORATORIO[\s\w])"
        ex_istituto = r"([\s\w]*ISTITUTO\s\w])"
        ex_clinic = r"[\s\w]+OF [\s\w]?(CLINIC (?:OF|FOR)[\s\w]*[\w]+)|"
        ex_clinic += r"[\s\w]+(?:OF|FOR) ([\s\w]+?CLINIC)|"
        ex_clinic += r" ?([\w]+[\s\w]*CLINIC(?: (?:OF|FOR)[\s\w]*[\w]+)?)|"
        ex_clinic += r"( ?CLINIC (?:OF|FOR)[\s\w]+)|"
        ex_clinic += r"([\s\w]*CLINIQUE[\s\w]*)"

        ex_centre = r"([\s\w]*CENTR(?:E|O)[\s\w]*)"
        ex_inc = r"([\s\w\.]+ INC(?:\.|,| ))"
        ex_station = r"[\s\w]?OF [\s\w]?(STATION (?:OF|FOR)[\s\w]*[\w]+)|"
        ex_station += r"[\s\w]+OF ([\s\w]+?STATION)|"
        ex_station += r"([\w]+[\s\w]*STATION(?: (?:OF|FOR)[\s\w]*[\w]+)?)|"
        ex_station += r"(STATION (?:OF|FOR)[\s\w]+)"
        ex_office = r"([\s\w]+ OFFICE)"

        type1 = 'edu'  # 大学，学校...==学术界
        regex_edu1 = ex_university + "|" + ex_univ + "|" + ex_university_chinese
        regex_edu2 = ex_college + "|" + ex_coll
        regex_edu3 = ex_school + "|" + ex_school_chinese

        type2 = 'com'  # 公司、集团、企业...==工业界
        regex_com1 = ex_group + "|" + ex_group_chinese
        regex_com2 = ex_corporation + "|" + ex_corporation_chinese + "|" + ex_inc
        regex_com3 = ex_factory + "|" + ex_factory_chinese

        type3 = 'org'  # 研究所，研究院...==科研机构
        regex_org1 = ex_organization + "|" + ex_institute_chinese + "|" + ex_academy + "|" + ex_academy_chinese
        regex_org2 = ex_institute + "|" + ex_inst + "|" + ex_institute_chinese + "|" + ex_institut + "|" + ex_istituto
        regex_org3 = ex_scenter + "|" + ex_scenter_chinese + "|" + ex_centre
        regex_org4 = ex_laboratory + "|" + ex_laboratory_chinese + "|" + ex_search_chinese + "|" + ex_laboratoire

        type4 = 'gov'  # 政府部门
        regex_gov1 = ex_hospital + "|" + ex_hospital_chinese + "|" + ex_bureau + "|" + ex_bureau_chinese + "|" + ex_ministry\
                     + "|" + ex_observatory + "|" + ex_kexie
        regex_gov2 = ex_clinic + "|" + ex_clinic_chinese + "|" + ex_center + "|" + ex_center_chinese + "|" + ex_committee\
                     + "|" + ex_commission + "|" + ex_ministry_chinese + "|" + ex_service+ex_ministry \
                     + "|" + ex_foundation + "|" + ex_council + "|" + ex_fund+ex_project
        regex_gov3 = ex_gov_chinese + "|" + ex_newspaper_chinese + "|" + ex_station\
                     + "|" + ex_office
        regex_gov4 = ex_agency + "|" + ex_institution_chinese + "|" + ex_da_group + "|" + ex_dang

        list1 = [(type1, regex_edu1), (type2, regex_com1), (type3, regex_org1), (type4, regex_gov1),
                 (type1, regex_edu2), (type2, regex_com2), (type3, regex_org2), (type4, regex_gov2),
                 (type1, regex_edu3), (type2, regex_com3), (type3, regex_org3), (type4, regex_gov3),
                 (type3, regex_org4), (type4, regex_gov4)]
        return list1

    def regex2(self):
        type1 = 'edu'  # 大学，学校...==学术界
        ex_university = r"([\s\w]*UNIVERSITY)|"
        ex_university += r"([\s\w]*UNIVERSITY)|"
        ex_university += r"([\s\w]*\-UNIVERSITY)|"
        ex_school = r"([\s\w]*SCHOOL)"

        regex_edu1 = ex_university

        type2 = 'com'  # 公司、集团、企业...==工业界


        type3 = 'org'  # 研究所，研究院...==科研机构
        ex_institute = r"([\s\w]*INSTITUTE)|"
        ex_institute += r"([\s\w]*INSTITUT)"
        ex_research = r"([\s\w]*RESEARCH)"
        ex_center = r"([\s\w]*CENTER)"
        ex_lab = r"([\s\w]*LABORATORY)|"
        ex_lab += r"([\s\w]*LABORATOIRE)"

        type4 = 'gov'  # 政府部门
        ex_department = r"([\s\w]*DEPARTMENT)"
        ex_division = r"([\s\w]*DIVISION)"

        list1 = [(type1, regex_edu1)]
        return list1

    def find_one_institute(self, aff):
        list_regex = self.regex()
        org = aff
        ins_type = 'none'
        flag = False
        if aff.strip() != "":
            for r in list_regex:
                if flag:
                    break
                rgx = r[1]
                reg = re.compile(rgx)
                result = reg.findall(aff)
                if len(result) != 0:
                    for i in range(len(result)):
                        org1 = "".join(result[i]).strip()
                        if org1 != '' and len(org1) >= 3:  # 只选择其中的一个
                            org = org1
                            ins_type = r[0]
                            flag = True
                            break
        return ins_type, org, flag

    def muti_language(self, affiliation):
        pass

    def weak_classify(self, affiliation):
        pass

    def find_inst(self, affiliation):
        (ins_type, org, flag) = self.find_one_institute(affiliation)
        if not flag:  # 这个是没有找到的情况
            affiliation = self.trimEmail(affiliation)  # 去掉一些无关紧要的部分
            affiliations = re.split('[&:|]', affiliation)
            if len(affiliations) > 1:
                for u in affiliations:
                    if u.strip() != '':
                        (ins_type, org, flag) = self.find_one_institute(u.strip())
                        if flag:
                            break
        # if not flag:
        #     self.weak_classify()
        # if not flag:  # 还是没有找到就要考虑多语言的问题
        #     self.muti_language(affiliation)
        return ins_type, org


if __name__ == '__main__':
    dest=u'LIMES (LIFE AND MEDICAL SCIENCES INSTITUTE), MOLECULAR GENETICS, UNIVERSITY OF BONN, BONN, GERMANY'
    # dest2 = 'SDFF.sfsf|ddd:eeee@163.com &sdasdas'
    dest = 'DEPARTMENT OF PATHOLOGY,IWATE MEDICAL UNIVERSITY SCHOOL OF MEDICINE,MORIOKA,JAPAN'
    an = addressNormalization()
    a = an.find_inst(dest)
    print(a, '####')
    print(len("".split(",")))
