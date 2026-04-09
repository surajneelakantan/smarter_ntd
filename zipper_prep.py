#!/usr/bin/env python3

import json

from numpy import nan
import pandas as pd

import cli_streamlit_compat


COURSE_FILE_ROOT = "../course_material"
COURSE_MANIFEST_DIR = "../course_zip_jsons"


def prepare_path2arcname(df_course: pd.DataFrame) -> dict[str, str]:
    path2arcname = {}
    for _, row in df_course.iterrows():
        dirname = row['course_name']
        for filename in row['pdf_name'], row['video_related_to_pdf']:
            if filename and not pd.isna(filename):
                arcname = f'{dirname}/{filename}'
                path = f'{COURSE_FILE_ROOT}/{arcname}'
                path2arcname[path] = arcname
    return path2arcname


def prepare_course_json(df_course: pd.DataFrame, session_id: str) -> str:
    path2arcname = prepare_path2arcname(df_course)
    with open(f'{COURSE_MANIFEST_DIR}/course-{session_id}.json', 'w') as file:
        json.dump({'path2arcname': path2arcname}, file,
                  indent=4, ensure_ascii=False)
    return f'/zip/course-{session_id}.zip'


def make_df_course() -> pd.DataFrame:
    return pd.DataFrame(
        [{'content_type': 'misc',
          'course_name': 'DT723A - Machine Learning',
          'has_video': False,
          'key_words_pdf': 'Machine Learning, Agenda, Introduction, Feature '
                           'Extraction, Selection, GMM, HMM, DBN, Anomaly Detection, '
                           'Confusion Matrix, Bacteria Identification, Blood Culture, '
                           'Deep Learning, Unsupervised Feature Learning, Medical '
                           'Applications, Automatic Kidney Stone Detection',
          'key_words_video': nan,
          'number of pages': 66,
          'pdf_duration': 99.0,
          'pdf_name': 'Smarter_TeacherIntro_H2024.pdf',
          'pdf_summary': 'The Machine Learning course covers introductory topics, '
                         'including course content and administration, followed by '
                         'discussions on machine learning applications and interests. '
                         'The course aims to provide knowledge and understanding of '
                         'fundamental concepts in machine learning, different classes '
                         'of algorithms, and ways to evaluate performance. '
                         'Additionally, the course introduces various aspects of '
                         'machine learning, including linear algebra, basic '
                         'statistics, optimization basics, and programming. Students '
                         'are encouraged to bring their own data for a final project, '
                         'where they will present their findings in a seminar.\n'
                         '\n'
                         'Here are 3-5 learning objectives:\n'
                         '\n'
                         '• Understand the fundamental concepts in machine learning '
                         'and its applications.\n'
                         '• Identify different classes of machine learning algorithms '
                         'and evaluate their performance.\n'
                         '• Apply mathematical concepts and programming skills to '
                         'solve machine learning problems.\n'
                         '• Develop critical thinking and problem-solving skills '
                         'through practical experience with real-world data.\n'
                         '• Present research results effectively to an audience.',
          'video_duration': nan,
          'video_duration_min': 0.0,
          'video_file_name': nan,
          'video_related_to_pdf': nan,
          'video_transcript': nan,
          'video_transcription_summary': ''},
         {'content_type': 'misc',
          'course_name': 'DT712A - Natural Language Processing',
          'has_video': True,
          'key_words_pdf': 'Taxonomy, Learning, Joint, NLP, Course, ML, Regression, '
                           'Clustering, Classification, Neural Network, NLP, '
                           'Supervised, Unsupervised, Semi-supervised, Reinforcement '
                           'learning, Dataset, Labels, Input, Output',
          'key_words_video': nan,
          'number of pages': 14,
          'pdf_duration': 21.0,
          'pdf_name': 'M4_1B_LearningTaxonomy.pdf',
          'pdf_summary': 'Here is a summary of the excerpt:\n'
                         '\n'
                         'The main topic is an introduction to Machine Learning (ML) '
                         'taxonomy and famous tasks in ML. The text discusses '
                         'different types of learning, including supervised, '
                         'unsupervised, semi-supervised, and reinforcement learning. '
                         'It also mentions three famous tasks in ML: regression, '
                         'clustering, and classification.\n'
                         '\n'
                         'Learning Objectives:\n'
                         '\n'
                         '• Identify the different types of machine learning '
                         '(supervised, unsupervised, semi-supervised, and '
                         'reinforcement learning).\n'
                         '• Recognize the three famous tasks in machine learning '
                         '(regression, clustering, and classification).\n'
                         '• Understand the concept of neural networks for NLP.\n'
                         '• Explain the differences between supervised and '
                         'unsupervised learning.',
          'video_duration': nan,
          'video_duration_min': 0.0,
          'video_file_name': nan,
          'video_related_to_pdf': 'M4_1B_LearningTaxonomy (Source).mp4',
          'video_transcript': nan,
          'video_transcription_summary': ''},
         {'content_type': 'misc',
          'course_name': 'DT119U - Data Mining',
          'has_video': False,
          'key_words_pdf': 'Python, Jupyter Notebook, Google Drive, Colab, Blackboard, '
                           'Assignments, Tasks, Programming',
          'key_words_video': nan,
          'number of pages': 8,
          'pdf_duration': 12.0,
          'pdf_name': 'HT24, Guideline to Prepare for Assignments.pdf',
          'pdf_summary': 'Here is a summary of the excerpt:\n'
                         '\n'
                         '**Main Topic:** Guidelines for Preparing Assignments (HT24)\n'
                         '\n'
                         'The document explains how to prepare and submit assignments '
                         'A1 to A4. Each assignment is a Jupyter Notebook file that '
                         'includes explanation, guidelines, Python codes, and '
                         'questions. Students need to download the files from '
                         'Blackboard, upload them to their Google Drive, perform '
                         'tasks, and share the files with the instructor.\n'
                         '\n'
                         '**Learning Objectives:**\n'
                         '\n'
                         '* Understand how to prepare and submit assignments A1 to A4\n'
                         '* Learn how to use Jupyter Notebook files and Google Colab\n'
                         '* Familiarize yourself with Python programming and packages '
                         'for data mining\n'
                         '* Know how to work with data files on your Google Drive',
          'video_duration': nan,
          'video_duration_min': 0.0,
          'video_file_name': nan,
          'video_related_to_pdf': nan,
          'video_transcript': nan,
          'video_transcription_summary': ''},
         {'content_type': 'misc',
          'course_name': 'DT717A - Autonomous Robots and ROS',
          'has_video': True,
          'key_words_pdf': 'Autonomous, Robots, ROS, Introduction, Project, '
                           'Simulation, Visualization',
          'key_words_video': 'messages package, ross basically, master communication, '
                             'concepts used, ss interfaces, program viewers, different '
                             'terminals, converter sense, compile lists, rough texture',
          'number of pages': 22,
          'pdf_duration': 33.0,
          'pdf_name': '0_Course_overview.pdf',
          'pdf_summary': 'Here is a summary of the excerpt:\n'
                         '\n'
                         '**Main Topic:** Autonomous Robots and ROS (Robot Operating '
                         'System) Course Overview\n'
                         '\n'
                         'The course consists of two parts: Introduction to ROS and '
                         'Project Work. The students will learn about nodes, messages, '
                         'and topics in ROS and complete small assignments before '
                         'working on their own projects.\n'
                         '\n'
                         '**Learning Objectives:**\n'
                         '\n'
                         '* Understand the basics of ROS and its components\n'
                         '* Learn how to use online resources (tutorials, pdfs, '
                         'YouTube) for learning ROS\n'
                         '* Develop problem-solving skills by completing small '
                         'assignments\n'
                         '* Apply ROS to a real-world project or scenario\n'
                         '\n'
                         'Note: The provided text does not contain any specific '
                         'learning objectives related to autonomous robots.',
          'video_duration': '15:12',
          'video_duration_min': 15.2,
          'video_file_name': '4_concepts.mp4',
          'video_related_to_pdf': '4_concepts.mp4',
          'video_transcript': 'okay so in this lecture with talk about different '
                              'concepts used in ross such as topics messages notes and '
                              'other vocabulary and if we look at ross ross is '
                              'basically a set of of different software tools on small '
                              'software binaries and what especially useful before his '
                              'to send data between this small programs we have a it '
                              'also very good for model i think and one key aspect '
                              'here is that it uses simpler clean and messages so is '
                              'that if you have a package just doing something that '
                              'you avoid having any dependents this between packages '
                              'so instead of that they have to share data types that '
                              'were defined in and other library for example yeah only '
                              'work with what we called simple are very clean messages '
                              'that say yeah like kicking tigres or strings or '
                              "something like that so you don't have to have any any "
                              "and independence's really between packages for this is "
                              'of course very nice you can see this sort of us away to '
                              'to force or the nazis or ross programs tease messages '
                              "ss interfaces so if you look at ross this to let's say "
                              'of having worry that is used in order to handle this '
                              'messages are sent is called raw score you can think of '
                              'this is a master some sort of master of communication '
                              'so for example you can send something through this raw '
                              'score topic and the topic comes with the message type '
                              'so that could be something we call a publisher so a '
                              'small program publish a message type on on a topic and '
                              "then we'll have also some subscribers somebody some "
                              'software some small program that as subscribe to this '
                              'and received this type of of message so for want to '
                              'make a very simple example here the start by starting '
                              'three different terminals can clear things up here bit '
                              'so in order to start mortar most used press control '
                              'shift and t and there we have a third term so and the '
                              "first one we start over think it's wrong role score so "
                              'now we have started than master in the second terminals '
                              'with tie in something called her off topic echo echo '
                              'something that typically sprint something on the screen '
                              'so and now have to specify what is called the topic '
                              "let's and this in this case cause magical number and "
                              "now it says that it doesn't appear to be published yet "
                              'which is very strong in this case and now iran '
                              "something called rostrum i'm sorry last topic cop and "
                              'here we specify what type of the topic we want to '
                              'published on we also need to specify what type of '
                              'message sees than the rest of houston and messages we '
                              'can use here for example an int with it fits i hear '
                              'this tab come in handy also because you can also '
                              'specify what if your tab you can see what type of date '
                              "that this message this don't have and this case is only "
                              'have a data type a message where we can specify '
                              'something so if we specify for the to here and press '
                              'enter if we now have a back to this terminal to we now '
                              'have received this data packets the two okay so this '
                              'was a a very basic example is to show show the main '
                              'principles here ah so recap here we have the master we '
                              'have the subscriber and we have the publisher saw the '
                              'subscriber listen to topic the publisher publish and we '
                              'also have to specify something which call topic and '
                              "that sort of it's a channel and name of a channel where "
                              'you can listen it and subscribe to and along with this '
                              'one if you publish something he also have to specify '
                              'the message tie in this case this echo is quite generic '
                              'so it can listen to all possible different type of '
                              'message type and and print quite a few of them out is '
                              'on the screen so if want to have another example so '
                              "let's see for can for example use the the web camera on "
                              'the laptop and in order to do that we need to have some '
                              'sort of driver software so we need to download some '
                              'some cameron drivers and here we will you go to our '
                              'smarter workspace into the source director so at the '
                              'moment here we all have our smarter or smith the '
                              'package so he is type youth close top and use the can '
                              "and now it's downloading it and now we have something "
                              'was called you speak am now the packets here and in '
                              'order to compile lists one we need to go to our smarter '
                              'workspace am rom can make and how it should be able to '
                              'build some stuff here which is related to something '
                              "called you speak camp for example so here's building a "
                              'library and and a binary so the or concepts here is '
                              'that we have our camera driver this driver republish '
                              'something on a topic here in this case you become image '
                              'role and it has a specific censor message this is then '
                              'pass through there raw score the communication master '
                              'and to not a program with the listen to this topic so '
                              'here we have the publisher and the subscriber so if '
                              'want to run this set up we can try it out using ross '
                              'launches v can and news be chem test so here we see the '
                              'example of their image viewer so this is that the '
                              'program that then listen to the data that is generated '
                              'by the camera driver here can for example see some '
                              "output i mean the camera driver doesn't have it's own "
                              'on giving him but he can see for example what kind of '
                              'device you think so here is using this dev video see or '
                              'hear can see the resolution six hundred forty for '
                              "eighty and you can also see the the frequency it's "
                              'eight six right so what if we then is that add another '
                              'image few here another subscribers so in principle i '
                              'mean this is their communication master it should also '
                              "be able to provide data to this two and let's try to "
                              'run that so have smarter image processing package dual '
                              'viewers so what this than those that the to bring up to '
                              'have this years so stanley will now one driver running '
                              'but we in this case has two different years how cake so '
                              'what if we then do a bit more things here so instead of '
                              'having this image viewer subscriber we have something '
                              'we call hear another note a call to grayscale converter '
                              "which then sense out something else it's still there "
                              "censor message image or like a message time but it's "
                              'using image great topic instead and then what if we '
                              'then fed this image great topic came to the image '
                              'viewer what is interesting here is that this one '
                              'actually is both a subscriber and the publisher and '
                              "this is perfectly fine to be it's so it's subscribe to "
                              'this use be camp a mr or topic and it published on this '
                              'image great topic so if want around this set up we '
                              'instead from this program so your viewers gray and '
                              "that's also another linux thing if used vs press and "
                              'the arrow up then you will get the last command you he '
                              'used so for on this setup is that what you can see here '
                              'is that will have two images so one with color and '
                              'wrong with and grayscale saying this case the image '
                              'viewer to within the image okay so what we can also do '
                              'here is that we used have we can peek a bit than have a '
                              'look into the cold so if he is go to our smarter image '
                              'processing here we have sources and here we have '
                              'actually our note so used for you to just see a bit how '
                              'how to cook called look like it can see it here so what '
                              'we have here is used to main we create our subscriber '
                              "so we can see that here's the subscriber know we're "
                              'looking into this box we have a use be can image draw '
                              'subscriber and here we basically says as soon as you '
                              'get the message here on this topic run this image call '
                              'back and nothing we define the down here he can also '
                              'see a few comments here so have a look at this filing '
                              "go go through it he'll have something which we called "
                              'advertise so in this sale we say probably something on '
                              'this topic image great so we make our an instance of a '
                              'publisher class in this case could pump so if we look '
                              'at this image call back what is simply thus is that '
                              'along with this one that topic we have the the message '
                              "type which is this call the sensitive message in it's "
                              'what you simply do here is that we convert that into '
                              'another format and we use this this is an open cd '
                              'format we just use the standard command here that '
                              'commerce into grayscale and then we simply publish this '
                              'message house and up that so this is the only thing i '
                              'actually running in this country right so now i draw '
                              'this things in in in powerpoint simpler but there are '
                              'also tools in rough texture draw something similar here '
                              'so this is output of something called are kitty kitty '
                              'and graph and here can see the use be can for example '
                              'would correspond to the camera driver the grayscale '
                              'converter would correspond to this boxer image viewer '
                              'to have over here and image you here and you can see '
                              'also the flow hear that the use be calm image throw '
                              'here is something which is some directed to this image '
                              'few and as well to this grayscale converter so if we '
                              'slice the the current so we have to start start up '
                              'again ah we can run this or kitty graph and here should '
                              'be able to see now the different and here can see that '
                              'the use be is providing data to the and the grayscale '
                              'converter and as well as the image you and your we can '
                              'see also that this grace can convert event of a '
                              'provides data to this image few number two okay so what '
                              'we have here is that these an elliptical things are '
                              "what we called notes in the ross weren't so everything "
                              'all this more programs that the either subscribed or '
                              'publish or a combination of the to like the grayscale '
                              'converter is something which is called a note ross note '
                              "and if you don't use this are kitty and to we can of "
                              'course use the command line so so this on so for '
                              'example last known list would be a tool that to display '
                              'old notes that the running in system so you can for '
                              'example see the the camera driver we have here the '
                              'image viewers we have also grayscale market we have '
                              'also some other stuff this is just used to non order to '
                              'to print things and we also have this our duty agree '
                              'this is actually also and out the thing that we used in '
                              'order to show the graph so basically this one is also '
                              'note in in in their offspring or if we want to check '
                              'all the available topics we can rom trust topic list '
                              'and see a list of all different topics if you on '
                              'suspicion information about the some note iran ross '
                              'known info and them for example this case then '
                              'grayscale converter how we can have a look here what '
                              "type of subscription it thus so it's a script "
                              'subscribed to this topic with this message and it '
                              'published on this topic with business and i think that '
                              'can maybe be interesting if to check how often is '
                              'thanks sent over topic sofia use ross topic hurts and '
                              'this image great but will get some statistics or how '
                              'many times this this populist with some statistics and '
                              'for example if you want to have some idea of what type '
                              'of message type is one topic you sing so for example '
                              'got tight mrs type to have him this image spray can '
                              'check this by typing this comment okay that was ever '
                              'thing about notes topics messages than and so thank you',
          'video_transcription_summary': 'Rss is a set of of different software tools '
                                         'on small software binaries and what '
                                         'especially useful before his to send data '
                                         'between this small programs. One key aspect '
                                         'here is that it uses simpler clean and '
                                         'messages so is that if you have a package '
                                         'just doing something you avoid having any '
                                         'dependents this between packages.'},
         {'content_type': 'misc',
          'course_name': 'DT717A - Autonomous Robots and ROS',
          'has_video': True,
          'key_words_pdf': 'ROS, package, code, compile, run, hello world, tutorial',
          'key_words_video': 'linux tutorial, copy director, build things, workspace '
                             'order, different packages, combine binary, specify '
                             'footpath, way, names used, packets degenerated',
          'number of pages': 16,
          'pdf_duration': 24.0,
          'pdf_name': '3_HelloWorld_ROS.pdf',
          'pdf_summary': 'Here is a summary of the excerpt:\n'
                         '\n'
                         '**Main Topic:** Creating a "Hello World" package in ROS '
                         '(Robot Operating System)\n'
                         '\n'
                         '**Summary:** The excerpt provides step-by-step instructions '
                         'on how to create and run a "Hello World" program in ROS. It '
                         'covers downloading code, compiling, running, and creating a '
                         'new package.\n'
                         '\n'
                         '**Learning Objectives:**\n'
                         '\n'
                         '* Download the Smarter_ROS code base and place it in the '
                         'correct directory\n'
                         '* Compile and run a simple "Hello World" program in ROS\n'
                         '* Create a new package using catkin_create_pkg\n'
                         '* Modify the CMakeLists.txt file to build the sources and '
                         'describe dependencies\n'
                         '\n'
                         'Note that these learning objectives are based solely on the '
                         'provided text and may not be comprehensive or representative '
                         'of the full scope of the topic.',
          'video_duration': '10:20',
          'video_duration_min': 10.333333333333334,
          'video_file_name': '3_hello_world.mp4',
          'video_related_to_pdf': '3_hello_world.mp4',
          'video_transcript': 'gay in this lecture will start doing our first many '
                              "projects for many package has a corner us and it's "
                              'called hello world so in this couple of lights will '
                              'will take how to download some code and where we should '
                              'place to coat and then how to compile and how to run a '
                              'very simple program so first thing is where the right '
                              'place my code and everything has to go into this '
                              'smarter workspace source three three so now we are at '
                              "this they're smarter workspace directory and we have to "
                              'going to sources and how the same directory there is of '
                              'course many different ways have to do it and so i think '
                              "if you're not familiar with linux you can also try to "
                              'follow linux tutorial on on some basic months we will '
                              'do a lot of things in this command prompt so i think '
                              'they could be well invested time in order to clone or '
                              'to download software police something called get here '
                              'so i put the bit of code here on get up and so this '
                              'eight seventy five seventy eight seventy five just my '
                              "username so that's why away where that comes from so "
                              'and now we have down on the code is not a huge amount '
                              'the code and in order to build that we always have to '
                              'be in the following directory in the smarter and '
                              'workspace three three and then in order to build things '
                              'we is tight cat can make and also within this is not '
                              "only this small hello or project it's also bit of other "
                              'things that we will use in in in later lectures okay so '
                              'know everything was sir can fight so we can now start '
                              'by running this small program and interests you can use '
                              'this command to run things so if you use type are '
                              'awesome smarter hello world and smarter hello world and '
                              'there is our hello world output so if you start looking '
                              'at where are our files located so we have this mater '
                              'workspace and in this one will have the source and here '
                              'we have our smarter or so this was the one thing we '
                              "download and now so if we're going to this one this "
                              'motorists is something called me at the packets and '
                              "it's just basically an empty package you can say but "
                              'where you have place to put smaller packages or '
                              'packages that belonged to each other says nice way to '
                              'structure director so within this mater us we have also '
                              'some other example coast here but now we were looking '
                              'to be smarter hello and within this code we have now '
                              'these different files so the see make least his '
                              'description of how to build coat so fewest look at it '
                              'quickly then he can see that this mainly comments here '
                              'on so this is say when you create the packets this '
                              'would be out degenerated and he can use this comments '
                              'in order to to have a look i know where to place things '
                              'and and how how to write things at the same thing is '
                              'also the ross framework is that in order to describe '
                              'the dependencies between other packages so this mater '
                              'hello world is something which a call a package of are '
                              "small package and also says this doesn't really have "
                              'any dependencies this is also mainly comments and '
                              "doesn't really container thing then of course in the "
                              'source folder we have also that should soft or that '
                              "were on and yeah well it's not much it's use this this "
                              'point okay so what about the binary star i mean if we '
                              'combine something then the binary have to end up '
                              'somewhere and in ross all the sources and all the '
                              'binary are that in two different packages but if we '
                              'want to find where this smarter hello world program is '
                              'he can find it here for is they seek it and if you on '
                              'the on it you can also specify the footpath and this is '
                              'that with again but it was nice thing here is that you '
                              "don't really have to keep track of anything see a "
                              'strong rostrum the name of the package smarter hello '
                              "aren't and then smarter alert is actually the name of "
                              'of the binary and they go so within the roster is a few '
                              'ah thanks to help you sort of navigate through through '
                              'this type of file system so for example if you want to '
                              'go to this mocked her hello world director for example '
                              'they suck man called raw see the so free go to the '
                              "router them to to my home directory and then we're "
                              'going to want to go directly to the smarter hello '
                              "aren't we can just type ross cd and then smarter and "
                              'then there is something called tab completion that is '
                              'in general used in in the command prompting the next '
                              "heavily but it's just that type of you letter and then "
                              'it will sort of in everything that fits your sofa in '
                              'this case we have a few smarter packages but there '
                              "should be enough that you're piss hates and time and "
                              'time again then it should automatically find out that '
                              "is the hello world it's go to and there ago so this is "
                              'a nice way instead of at navigating through with using '
                              'am quite lengthy and like thought for a directory names '
                              'used to use this atrocity for example of that would as '
                              'speed up things quite of it so if we now look into how '
                              "do we create the packet so let's create the many "
                              'another hello world package and if you want some more '
                              'detail can you can find on on on this web page and an '
                              'example of a very small packages this smarter hello '
                              'world and again this motorists would be a method '
                              "packets it's not required but it's a it's a nice way to "
                              'sort of cluster or put things similar packets that '
                              'belongs to the similar project or a similar type of '
                              'robots or something like that into into the same '
                              'package so if we want to create a new package this '
                              'smarter test and we should create that that is said '
                              'here in the smarter oss packets we need to go to the '
                              'smarter us and we simply type pierre cat can create '
                              'package and then the name of the fact that you have to '
                              'create so grid test and here we know how something '
                              'which is motors call smarter smarter tests and within '
                              'this smarter tests would have a few fights or lady '
                              'generated the face with describes how to build things '
                              'and then the package from so if we sorry if you go down '
                              'into this here is the find that was out the generated '
                              'the see make list file and the package and if you can '
                              'also have a look at them of course and the same thing '
                              "as with the the smarter and hello packets it doesn't "
                              'contain and independents this release on both his '
                              "fights is set it's main an empty but in order to have "
                              'that leaves something to cook comply hilarious take '
                              'their smarter hello world program and then yes copy it '
                              'to to this director so games this or see serious marker '
                              'test now actually are already in this folder as you see '
                              'l the father is completely empty so we have to make a '
                              'source directors or have somewhere to place our sources '
                              'and then we just copy the smarter hello world program '
                              'to the source director okay but in order to get '
                              'everything to run we also of course need to change this '
                              'see make list file so we just have to open it so he is '
                              'he had it said small nice editor three can use you want '
                              'and we need to sort of place this line somewhere within '
                              'the coat and there are some field section here where we '
                              'can place it so for have some forget and choice so at '
                              'executable smarter tests hello world and here we have '
                              'at i the source fight so messy a take a source filing '
                              'pilot thing too smart the test hello world so should '
                              'just say this one control as and then we have to build '
                              'and again as soon as the hey you you need to build '
                              'anything you have to be in this marked or workspace '
                              'directory so kept can make and now we should see here '
                              "and out with that it's tryst build something here folks "
                              'on yes saw smarter test hello world and in order to run '
                              'this try out ross smart the test and smarter test hello '
                              'world and they would have it so that was a quick tour '
                              'about just how to create packages and that was it thank '
                              'you',
          'video_transcription_summary': 'In this lecture will start doing our first '
                                         'many projects for many package has a corner '
                                         "us and it's called hello world. Pierre cat "
                                         'can create package and then the name of the '
                                         'fact that you have to create so grid test. '
                                         'Within this smarter tests would have a few '
                                         'fights or lady generated the face with '
                                         'describes how to build things.'},
         {'content_type': 'slides',
          'course_name': 'DT712A - Natural Language Processing',
          'has_video': True,
          'key_words_pdf': 'Machine Learning, Underfitting, Overfitting, Training, '
                           'Validation, Test Sets, Generalization',
          'key_words_video': nan,
          'number of pages': 16,
          'pdf_duration': 24.0,
          'pdf_name': 'M3_1H_OverFittingUnderFitting.pdf',
          'pdf_summary': 'Here is a summary of the excerpt in ≤120-word paragraph:\n'
                         '\n'
                         'The main topic is underfitting and overfitting, which are '
                         'common issues that can occur when developing machine '
                         'learning models. Underfitting occurs when the model is too '
                         'simple and fails to represent the data well, making it '
                         'difficult to make accurate predictions on unseen test data. '
                         'On the other hand, overfitting occurs when the model is too '
                         'complex and has learned too much from the training data, '
                         'making it perform poorly on new, unseen data.\n'
                         '\n'
                         'Here are 3-5 bullet learning objectives:\n'
                         '\n'
                         '• Understand the concept of underfitting and its '
                         'implications for machine learning models.\n'
                         '• Recognize the signs of overfitting in a model and its '
                         'consequences.\n'
                         '• Learn how to handle underfitting by increasing the size or '
                         'number of parameters in the model, training with more data, '
                         'or increasing the training time.\n'
                         '• Discover techniques to prevent overfitting, such as early '
                         'stopping, regularization, and dropout.',
          'video_duration': nan,
          'video_duration_min': 0.0,
          'video_file_name': nan,
          'video_related_to_pdf': 'M3_1H (Source).mp4',
          'video_transcript': nan,
          'video_transcription_summary': ''}]
    )


def prepare_json_and_button(df_course: pd.DataFrame) -> None:
    session_id = input('<session_id>')
    url = prepare_course_json(df_course, session_id)
    cli_streamlit_compat.show_download_button("Download course materials", url)


def main():
    df_course = make_df_course()
    prepare_json_and_button(df_course)


if __name__ == '__main__':
    main()