import pstats
p = pstats.Stats('profile_output.prof')
p.sort_stats('cumulative').print_stats(50) # 例: 累積時間でソートし上位30件表示
# p.sort_stats('tottime').print_stats(30) # 例: 関数自体の実行時間でソート