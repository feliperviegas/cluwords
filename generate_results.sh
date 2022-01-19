words=20; for strategy in fasttext_wiki_original fasttext_wiki_bert_0_4 fasttext_wiki_bert_0_6 ; do for dataset in info ever face; do echo -n "${strategy} - ${dataset} TopWords (${words}): "; tail -1 "${strategy}/results/${dataset}/result_topic_${words}.txt"; done; done

