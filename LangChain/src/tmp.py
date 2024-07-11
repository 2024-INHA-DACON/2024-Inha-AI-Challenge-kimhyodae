for year, credit_dict in credits.items():
            writer.writerow({
                '학년': year,
                '교양 필수 총 학점': credit_dict['교필'],
                '교양 선택 총 학점': credit_dict['교선'],
                '전공 필수 총 학점': credit_dict['전필'],
                '전공 선택 총 학점': credit_dict['전선'],
                '일반 선택 총 학점': credit_dict['일선'],
                '교직 선택 총 학점': credit_dict['교직']
            })