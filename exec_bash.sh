
for dataset in acm ang drop ever face info pinter trip tweets uber wpp; do echo ${dataset}; done

top_words=5;for dataset in acm ang drop ever face info pinter trip tweets uber wpp; do tail -1 fasttext_wiki_baseline_0_4/results/${dataset}/result_topic_${top_words}.txt | sed 's/[()]//g' | awk '{print $3" "$4;}'; done



top_words=5;for dataset in acm ang drop ever face info pinter trip tweets uber wpp; do tail -1 fasttext_wiki_bert_concat_0_4/results/${dataset}/result_topic_${top_words}.txt | sed 's/[()]//g' | awk '{print $3" "$4;}'; done

