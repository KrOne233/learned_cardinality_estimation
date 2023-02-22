SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi.info_type_id>15;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year<1999 AND mc.company_type_id=2 AND mi_idx.info_type_id=100;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND t.production_year<2012 AND mc.company_id>7851 AND mi.info_type_id>16 AND mi_idx.info_type_id=101;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi.info_type_id>4;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND t.production_year=1992 AND ci.person_id<1816925;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2002 AND mi.info_type_id>1;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND ci.role_id>1;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND ci.person_id<926305 AND ci.role_id<10 AND mi.info_type_id<7 AND mk.keyword_id>1074;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND t.production_year=2004 AND ci.role_id<2 AND mk.keyword_id<2909;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>1999 AND mc.company_type_id=2 AND mk.keyword_id=13886;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=1 AND t.production_year<2008 AND mi.info_type_id<2 AND mk.keyword_id<3291;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND mc.company_id<428 AND ci.person_id<525577 AND ci.role_id=9;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=3 AND mc.company_id>19054 AND mc.company_type_id=2 AND ci.person_id>1108086 AND ci.role_id<10 AND mk.keyword_id=3644;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=99;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>1936 AND ci.person_id>1487650 AND ci.role_id<8 AND mk.keyword_id=16264;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi.info_type_id<8 AND mi_idx.info_type_id>100;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id<2 AND mc.company_id>80011 AND ci.person_id=613664 AND ci.role_id=1;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<6;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND ci.person_id=2029088 AND mi.info_type_id>13 AND mi_idx.info_type_id=99;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year<2011 AND mc.company_id>1060 AND mk.keyword_id<31445;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id<5569 AND mc.company_type_id=1 AND mi.info_type_id>1 AND mi_idx.info_type_id=99 AND mk.keyword_id<10833;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<8;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND t.production_year>2004 AND mi_idx.info_type_id<101 AND mk.keyword_id<44523;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND t.production_year=1979 AND mc.company_id>8488 AND ci.person_id>1502694 AND ci.role_id>1 AND mi.info_type_id=8 AND mi_idx.info_type_id=101;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND ci.person_id<2894607 AND mi.info_type_id<2;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_type_id=1 AND mi_idx.info_type_id<101 AND mk.keyword_id=74852;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id<7;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND ci.person_id>813072 AND ci.role_id=1 AND mi_idx.info_type_id=99;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id>11203 AND mc.company_type_id=2 AND mi.info_type_id=8 AND mi_idx.info_type_id=101 AND mk.keyword_id=52288;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND mi.info_type_id<4;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND t.production_year<1953 AND ci.person_id>335383 AND ci.role_id=5 AND mi_idx.info_type_id>99 AND mk.keyword_id>6898;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id>34 AND mc.company_type_id<2 AND mi_idx.info_type_id>99 AND mk.keyword_id<335;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND mc.company_id<52087 AND ci.person_id<158834 AND ci.role_id>1;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_type_id=2 AND mi.info_type_id=8 AND mi_idx.info_type_id<100;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND ci.person_id<1015782 AND ci.role_id<2 AND mi.info_type_id<16;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id>33285 AND mc.company_type_id>1 AND mi.info_type_id<13 AND mk.keyword_id>3921;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id<3 AND mc.company_type_id<2 AND mk.keyword_id<1535;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id=26870 AND mi_idx.info_type_id>100 AND mk.keyword_id>12102;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND ci.person_id<463537 AND mi.info_type_id>16 AND mi_idx.info_type_id<101 AND mk.keyword_id<16037;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id>1 AND mc.company_type_id=2 AND ci.person_id>990535 AND ci.role_id=2 AND mi_idx.info_type_id=101;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year>2010 AND ci.person_id<614751 AND mi.info_type_id<16 AND mi_idx.info_type_id<100;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi.info_type_id>8 AND mi_idx.info_type_id<100 AND mk.keyword_id>825;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2006 AND mk.keyword_id>3588;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year<2000 AND mc.company_id<127160 AND mc.company_type_id<2 AND ci.person_id<3521904 AND mi.info_type_id<16 AND mk.keyword_id>2375;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id>2 AND ci.role_id=6;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year=1958 AND mc.company_id>12740;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND ci.person_id<4042059;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year<1973 AND ci.role_id<9 AND mi.info_type_id=6 AND mk.keyword_id<3736;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND t.production_year=2011 AND mi.info_type_id=16 AND mk.keyword_id>870;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2004;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<100 AND mk.keyword_id<91063;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id>11323 AND mc.company_type_id=2 AND mi.info_type_id>3 AND mk.keyword_id>4962;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id>525 AND mc.company_type_id<2 AND ci.role_id=1 AND mi_idx.info_type_id>100;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=1969 AND mc.company_id<115151 AND mi.info_type_id=16 AND mi_idx.info_type_id>101 AND mk.keyword_id>1614;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi.info_type_id>4 AND mi_idx.info_type_id<100 AND mk.keyword_id>2048;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=100;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND mc.company_id=1627 AND mc.company_type_id>1 AND mk.keyword_id>9986;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND mi.info_type_id=9 AND mi_idx.info_type_id<100 AND mk.keyword_id<71574;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year=2007 AND ci.role_id>1 AND mi.info_type_id>8 AND mi_idx.info_type_id<100;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year=2011 AND mi.info_type_id>15 AND mi_idx.info_type_id>100;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id=19 AND mc.company_type_id<2 AND mi.info_type_id<6 AND mk.keyword_id<6663;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mc.company_id=4482 AND mc.company_type_id<2 AND ci.person_id>1875226;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>1963 AND ci.role_id<5 AND mi.info_type_id<4;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND ci.person_id=360343 AND ci.role_id<2 AND mk.keyword_id<359;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND mc.company_type_id<2 AND mi.info_type_id<3;
SELECT COUNT(*) FROM title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi.info_type_id=16;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year=1997 AND mi_idx.info_type_id<101;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>19661 AND mc.company_type_id<2 AND mi.info_type_id<18 AND mi_idx.info_type_id>99;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.production_year=2012 AND mc.company_id=139 AND ci.role_id>10 AND mi.info_type_id=3 AND mi_idx.info_type_id=100;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mc.company_id<145 AND mc.company_type_id<2 AND ci.person_id>1174178 AND mi_idx.info_type_id<101;
SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND ci.person_id<2451133 AND mi_idx.info_type_id>99;
SELECT COUNT(*) FROM title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=1 AND mi.info_type_id<4;
SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2008 AND mi.info_type_id=8 AND mi_idx.info_type_id=101;
SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2009 AND mi.info_type_id=8 AND mi_idx.info_type_id=101;
SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2010;
SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2010 AND mc.company_id=22956;
SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000;
SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100;
SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100 AND t.production_year>2010;
SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2000 AND mi.info_type_id=8 AND mi_idx.info_type_id=101;
SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2005 AND mi.info_type_id=8 AND mi_idx.info_type_id=101;
SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000 AND t.production_year<2010 AND mk.keyword_id=7084;
SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000 AND t.production_year<2005 AND mk.keyword_id=7084;
SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100 AND t.production_year>2000;
