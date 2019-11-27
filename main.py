from SentenceSelector import select_sentences
text="""Mahatma Gandhi is well known as the “Father of the Nation or Bapu” because of his greatest contributions towards the independence of our country. He was the one who believed in the non-violence and unity of the people and brought spirituality in the Indian politics. He worked hard for the removal of the untouchability in the Indian society, upliftment of the backward classes in India, raised voice to develop villages for social development, inspired Indian people to use swadeshi goods and other social issues. He brought common people in front to participate in the national movement and inspired them to fight for their true freedom.

He was one of the persons who converted people’s dream of independence into truth a day through his noble ideals and supreme sacrifices. He is still remembered between us for his great works and major virtues such as non-violence, truth, love and fraternity. He was not born as great but he made himself great through his hard struggles and works. He was highly influenced by the life of the King Harischandra from the play titled as Raja Harischandra. After his schooling, he completed his law degree from England and began his career as a lawyer. He faced many difficulties in his life but continued walking as a great leader.

He started many mass movements like Non-cooperation movement in 1920, civil disobedience movement in 1930 and finally the Quit India Movement in 1942 all through the way of independence of India. After lots of struggles and works, independence of India was granted finally by the British Government. He was a very simple person who worked to remove the colour barrier and caste barrier. He also worked hard for removing the untouchability in the Indian society and named untouchables as “Harijan” means the people of God.

He was a great social reformer and Indian freedom fighter who died a day after completing his aim of life. He inspired Indian people for the manual labour and said that arrange all the resource ownself for living a simple life and becoming self-dependent. He started weaving cotton clothes through the use of Charakha in order to avoid the use of videshi goods and promote the use of Swadeshi goods among Indians.

He was a strong supporter of the agriculture and motivated people to do agriculture works. He was a spiritual man who brought spirituality to the Indian politics. He died in 1948 on 30th of January and his body was cremated at Raj Ghat, New Delhi. 30th of January is celebrated every year as the Martyr Day in India in order to pay homage to him.

"""
summary=select_sentences(text,20)
print(len(summary))
print(summary)