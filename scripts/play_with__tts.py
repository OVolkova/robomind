from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
import io

from robomind.speech_to_speech import TextToSpeech


tts = TextToSpeech()


def test_speech_to_speech(text):
    output_signal, output_frequency = tts.generate(text)

    memory_file = io.BytesIO()
    sf.write(memory_file, output_signal, output_frequency, format="wav")
    sound = AudioSegment.from_file(memory_file, format="wav")
    play(sound)


if __name__ == "__main__":
    t = """

Example One: Palestinian and Israeli Peace Efforts: Bereaved Families

There are many efforts among Israelis and Palestinians to forge peace but here we'll focus on those initiated by families who have lost loved ones in the conflict. Follow these links for a 7 minute extended trailer of a moving documentary entitled Encounter Point that was released in 2007, and the story of the Parents Circle-Family Forum. How can Galtung's typologies help us understand why these and related stories are not widely known?

Example Two: Thich Nhat Hanh and Engaged Buddhism

Thich Nhat Hanh is a Vietnamese Buddhist Monk who coined the term "Engaged Buddhism" during the Vietnam War to represent his active work for peace and to respond to the traumas of war. Please follow these links for  an overview of his life and teaching, a list of the fourteen precepts of Engaged Buddhism, and see below a video documenting his return from exile to Vietnam to found a monastery and how it was later banned by the Communist government. In reflecting on Thich Nhat Hanh's long service as monk, what is unique about his peacebuilding efforts and what are the sources of his inspiration to forge new pathways?  

Example Three: Liberation Theology, and the "Preferential Option for the Poor" in Latin America

Peruvian Roman Catholic priest Gustavo Gutierrez is known in many circles as the "father" of liberation theology which is a movement that emerged in Latin America in the 1960s and is associated with interpreting the Christian Gospels through the experience of the poor and marginalized. In 1981, Pope John Paul II named Joseph Ratzinger (later Pope Benedict XVI) as Cardinal-Prefect for the Congregation of the Doctrine of the Faith, an office charged with defending and affirming official Catholic doctrine.  In this role, Cardinal Ratzinger condemned liberation theology and accused it of having Marxist affiliations and inciting violence.  See the following links for an article about Gustavo Gutierrez and a recently published collection of his writings, a 2008 article highlighting the tensions between the Vatican under Pope Benedict and liberation theology, and an article describing how Pope Francis is more aligned with the tenets of liberation theology than his predecessors were.  How can Galtung's typologies help understand the emergence of and controversies surrounding liberation theology?  

Example Four:  1790 Exchange of Letters Between Jewish Leader Moses Seixas and President George Washington

Moses Seixas, was the Warden of the Hebrew Congregation in Newport, Rhode Island, now the Touro Synagogue. In 1790, the year that the new Constitution of the United States was ratified, President Washington visited Rhode Island and Mr. Seixas was one of the dignitaries selected to greet the new president. Mr Seixus penned a letter for the occasion that highlighted, in part,  how Jews had been "deprived as we heretofore have been of the invaluable rights of free Citizens..." but that they now look to the newly established Republic to be "a Government, which to bigotry gives no sanction [and] to persecution no assistance..." President Washington responded a few days later with a letter of his own that read, in part, "It is now no more that toleration is spoken of, as if it was by the indulgence of one class of people, that another enjoyed the exercise of their inherent natural rights."  He went on to assert that "For happily, the Government of the United States gives to bigotry no sanction, to persecution no assistance, requires only that they who live under its protection should demean themselves as good citizens..." Further information and full transcripts of both letters are available (courtesy of the George Washington Institute for Religious Freedom). Please reflect upon the social and cultural conditions that gave rise to this aspirational exchange.  

Example Five: The Passion Play at Oberammergau

Beginning in 1634 and with few interruptions, the people of the village of Oberammergau, Germany have staged a Passion Play depicting a dramatic rendition of the arrest, trial, and crucifixion of Jesus of Nazareth. According to legend, in 1633 the bubonic plague had come to the region and the townspeople vowed to regularly perform a Passion Play if they would be spared. It is the longest running continual performance of a Passion Play in the world. Over the past several decades, critics have asserted that the play is anti-Semitic. See here for an article by Professor Anna Lisa Ohm depicting the history of the play and here for a promotional video produced by the town. One of the questions raised in the article is whether a "good" (e.g., not anti-Semitic) passion play is possible. How might Galtung's typologies help us to better understand this controversy?

Example Six: The Sri Lankan Civil War (1983-2009)

This long conflict took a terrible toll on the Sri Lankan population. Read this case study published by the Berkeley Center for Religion, Peace, and World Affairs at Georgetown University that addresses the following questions: What are the historical origins of the conflict in Sri Lanka? How were domestic religious forces and identities involved? How important were international religious and political forces? What role did socioeconomic factors play? As you read, please reflect upon how Galtung's typologies can be applied to this conflict.  

Example Seven: Gandhi's Legacy

See this excerpt and read pages 15-27 for an overview of the three basic precepts of Gandhi's foundation for nonviolence: Satyagraha, or "Truth Force" in Joan Bondurant's classic 1958 study of Gandhi entitled Conquest of Violence. See this National Geographic story about some of ways that Gandhi's legacy endures in India today. From the perspective of Galtung's typologies, reflect upon how cultures of peace are cultivated and sustained over time, especially when they are countering more pervasive cultures of violence.  

Example Eight: Climate Change and Boko Haram in Nigeria

See this news article in Mother Jones magazine from 2014 showing how environmental degradation has many social and political consequences that have, in part, served to exacerbate inter-religious tensions and fueled support for Boko Haram.  



"""

    for x in [
        "DISCUSSION QUESTIONS",
    ]:
        t = t.replace(x, x.lower())
    t = t.replace("\n", " ")
    for ttt in t.split("."):
        for tttt in ttt.split("?"):
            if tttt.strip():
                test_speech_to_speech(tttt)
    # for x in ["DISCUSSION QUESTIONS,]:
    #     t = t.replace(x, x.lower())
    # for tt in t.split("\n\n"):
    #     for ttt in tt.split("."):
    #         for tttt in ttt.split("?"):
    #             if tttt.strip():
    #                 test_speech_to_speech(tttt)
