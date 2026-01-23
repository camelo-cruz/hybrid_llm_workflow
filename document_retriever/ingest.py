from langchain_community.document_loaders import RecursiveUrlLoader

loader = RecursiveUrlLoader(
    "https://www.ista.com/de/kontakt-service/mieter-oder-bewohner/faq/?_gl=1*2oht2a*_up*MQ..*_gs*MQ..&gclid=Cj0KCQiA1czLBhDhARIsAIEc7uj6vSh1JY8tpgQ8QwSDSSv59373TzcLhxiL31WwfD-rmNavVwug8isaApazEALw_wcB&gbraid=0AAAAACQHgtxRsrbU5_kJaj1XSjFx-Yog9"
    )

docs = loader.load()
docs[0].metadata