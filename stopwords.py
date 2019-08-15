# All stopwords are taken from spaCy

# coding: utf8
from __future__ import unicode_literals


BN = set("""
অতএব অথচ অথবা অনুযায়ী অনেক অনেকে অনেকেই অন্তত  অবধি অবশ্য অর্থাৎ অন্য অনুযায়ী অর্ধভাগে
আগামী আগে আগেই আছে আজ আদ্যভাগে আপনার আপনি আবার আমরা আমাকে আমাদের আমার  আমি আর আরও 
ইত্যাদি ইহা 
উচিত উনি উপর উপরে উত্তর
এ এঁদের এঁরা এই এক একই একজন একটা একটি  একবার একে এখন এখনও এখানে এখানেই এটা এসো
এটাই এটি এত এতটাই এতে এদের এবং এবার এমন এমনি এমনকি এর এরা এলো এস এসে 
ঐ 
ও ওঁদের ওঁর ওঁরা ওই ওকে ওখানে ওদের ওর ওরা 
কখনও কত কথা কবে কয়েক  কয়েকটি করছে করছেন করতে  করবে করবেন করলে কয়েক  কয়েকটি করিয়ে করিয়া করায়
করলেন করা করাই করায় করার করি করিতে করিয়া করিয়ে করে করেই করেছিলেন করেছে করেছেন করেন কাউকে 
কাছ কাছে কাজ কাজে কারও কারণ কি কিংবা কিছু কিছুই কিন্তু কী কে কেউ কেউই কেন কোন কোনও কোনো কেমনে কোটি
ক্ষেত্রে খুব 
গিয়ে গিয়েছে গুলি গেছে গেল গেলে গোটা গিয়ে গিয়েছে
চলে চান চায় চেয়ে চায় চেয়ে চার চালু চেষ্টা 
ছাড়া ছাড়াও ছিল ছিলেন ছাড়া ছাড়াও
জন জনকে জনের জন্য জন্যে জানতে জানা জানানো জানায়  জানিয়ে  জানিয়েছে জানায় জাানিয়ে জানিয়েছে
টি 
ঠিক 
তখন তত তথা তবু তবে তা তাঁকে তাঁদের তাঁর তাঁরা তাঁহারা তাই তাও তাকে তাতে তাদের তার তারপর তারা তারই তাহলে তাহা তাহাতে তাহার তিনই 
তিনি তিনিও তুমি তুলে তেমন তো তোমার তুই তোরা তোর তোমাদের তোদের
থাকবে থাকবেন থাকা থাকায় থাকে থাকেন থেকে থেকেই  থেকেও থাকায়
দিকে দিতে দিয়ে দিয়েছে দিয়েছেন দিলেন দিয়ে দু  দুটি  দুটো দেওয়া দেওয়ার দেখতে দেখা দেখে দেন দেয়  দেশের  
দ্বারা দিয়েছে দিয়েছেন দেয় দেওয়া দেওয়ার দিন দুই
ধরা ধরে 
নয় না নাই নাকি নাগাদ নানা নিজে নিজেই নিজেদের নিজের নিতে নিয়ে নিয়ে নেই নেওয়া নেওয়ার নয় নতুন
পক্ষে পর পরে পরেই পরেও পর্যন্ত পাওয়া পারি পারে পারেন পেয়ে প্রতি প্রভৃতি প্রায় পাওয়া পেয়ে প্রায় পাঁচ প্রথম প্রাথমিক
ফলে ফিরে ফের 
বছর বদলে বরং বলতে বলল বললেন বলা বলে বলেছেন বলেন  বসে বহু বা বাদে বার বিনা বিভিন্ন বিশেষ বিষয়টি বেশ ব্যবহার ব্যাপারে বক্তব্য বন বেশি
ভাবে  ভাবেই 
মত মতো মতোই মধ্যভাগে মধ্যে মধ্যেই  মধ্যেও মনে মাত্র মাধ্যমে মানুষ মানুষের মোট মোটেই মোদের মোর 
যখন যত যতটা যথেষ্ট যদি যদিও যা যাঁর যাঁরা যাওয়া  যাওয়ার যাকে যাচ্ছে যাতে যাদের যান যাবে যায় যার  যারা যায় যিনি যে যেখানে যেতে যেন 
যেমন 
রকম রয়েছে রাখা রেখে রয়েছে 
লক্ষ 
শুধু শুরু 
সাধারণ সামনে সঙ্গে সঙ্গেও সব সবার সমস্ত সম্প্রতি সময় সহ সহিত সাথে সুতরাং সে  সেই সেখান সেখানে  সেটা সেটাই সেটাও সেটি স্পষ্ট স্বয়ং 
হইতে হইবে হইয়া হওয়া হওয়ায় হওয়ার হচ্ছে হত হতে হতেই হন হবে হবেন হয় হয়তো হয়নি হয়ে হয়েই হয়েছিল হয়েছে হাজার
হয়েছেন হল হলে হলেই হলেও হলো হিসাবে হিসেবে হৈলে হোক হয় হয়ে হয়েছে হৈতে হইয়া  হয়েছিল হয়েছেন হয়নি হয়েই হয়তো হওয়া হওয়ার হওয়ায়
""".split())


DA = set("""
af aldrig alene alle allerede alligevel alt altid anden andet andre at

bag begge blandt blev blive bliver burde bør

da de dem den denne dens der derefter deres derfor derfra deri dermed derpå derved det dette dig din dine disse dog du

efter egen eller ellers en end endnu ene eneste enhver ens enten er et

flere flest fleste for foran fordi forrige fra få før først

gennem gjorde gjort god gør gøre gørende

ham han hans har havde have hel heller hen hende hendes henover her herefter heri hermed herpå hun hvad hvem hver hvilke hvilken hvilkes hvis hvor hvordan hvorefter hvorfor hvorfra hvorhen hvori hvorimod hvornår hvorved

i igen igennem ikke imellem imens imod ind indtil ingen intet

jeg jer jeres jo

kan kom kommer kun kunne

lad langs lav lave lavet lidt lige ligesom lille længere

man mange med meget mellem men mens mere mest mig min mindre mindst mine mit må måske

ned nemlig nogen nogensinde noget nogle nok nu ny nyt nær næste næsten

og også om omkring op os over overalt

på

samme sammen selv selvom senere ses siden sig sige skal skulle som stadig synes syntes så sådan således

temmelig tidligere til tilbage tit

ud uden udover under undtagen

var ved vi via vil ville vore vores vær være været

øvrigt
""".split())


DE = set("""
á a ab aber ach acht achte achten achter achtes ag alle allein allem allen
aller allerdings alles allgemeinen als also am an andere anderen andern anders
auch auf aus ausser außer ausserdem außerdem

bald bei beide beiden beim beispiel bekannt bereits besonders besser besten bin
bis bisher bist

da dabei dadurch dafür dagegen daher dahin dahinter damals damit danach daneben
dank dann daran darauf daraus darf darfst darin darüber darum darunter das
dasein daselbst dass daß dasselbe davon davor dazu dazwischen dein deine deinem
deiner dem dementsprechend demgegenüber demgemäss demgemäß demselben demzufolge
den denen denn denselben der deren derjenige derjenigen dermassen dermaßen
derselbe derselben des deshalb desselben dessen deswegen dich die diejenige
diejenigen dies diese dieselbe dieselben diesem diesen dieser dieses dir doch
dort drei drin dritte dritten dritter drittes du durch durchaus dürfen dürft
durfte durften

eben ebenso ehrlich eigen eigene eigenen eigener eigenes ein einander eine
einem einen einer eines einigeeinigen einiger einiges einmal einmaleins elf en
ende endlich entweder er erst erste ersten erster erstes es etwa etwas euch

früher fünf fünfte fünften fünfter fünftes für

gab ganz ganze ganzen ganzer ganzes gar gedurft gegen gegenüber gehabt gehen
geht gekannt gekonnt gemacht gemocht gemusst genug gerade gern gesagt geschweige
gewesen gewollt geworden gibt ging gleich gott gross groß grosse große grossen
großen grosser großer grosses großes gut gute guter gutes

habe haben habt hast hat hatte hätte hatten hätten heisst heißt her heute hier
hin hinter hoch

ich ihm ihn ihnen ihr ihre ihrem ihrer ihres im immer in indem infolgedessen
ins irgend ist

ja jahr jahre jahren je jede jedem jeden jeder jedermann jedermanns jedoch
jemand jemandem jemanden jene jenem jenen jener jenes jetzt

kam kann kannst kaum kein keine keinem keinen keiner kleine kleinen kleiner
kleines kommen kommt können könnt konnte könnte konnten kurz

lang lange leicht leider lieber los

machen macht machte mag magst man manche manchem manchen mancher manches mehr
mein meine meinem meinen meiner meines mensch menschen mich mir mit mittel
mochte möchte mochten mögen möglich mögt morgen muss muß müssen musst müsst
musste mussten

na nach nachdem nahm natürlich neben nein neue neuen neun neunte neunten neunter
neuntes nicht nichts nie niemand niemandem niemanden noch nun nur

ob oben oder offen oft ohne

recht rechte rechten rechter rechtes richtig rund

sagt sagte sah satt schlecht schon sechs sechste sechsten sechster sechstes
sehr sei seid seien sein seine seinem seinen seiner seines seit seitdem selbst
selbst sich sie sieben siebente siebenten siebenter siebentes siebte siebten
siebter siebtes sind so solang solche solchem solchen solcher solches soll
sollen sollte sollten sondern sonst sowie später statt

tag tage tagen tat teil tel trotzdem tun

über überhaupt übrigens uhr um und uns unser unsere unserer unter

vergangene vergangenen viel viele vielem vielen vielleicht vier vierte vierten
vierter viertes vom von vor

wahr während währenddem währenddessen wann war wäre waren wart warum was wegen
weil weit weiter weitere weiteren weiteres welche welchem welchen welcher
welches wem wen wenig wenige weniger weniges wenigstens wenn wer werde werden
werdet wessen wie wieder will willst wir wird wirklich wirst wo wohl wollen
wollt wollte wollten worden wurde würde wurden würden

zehn zehnte zehnten zehnter zehntes zeit zu zuerst zugleich zum zunächst zur
zurück zusammen zwanzig zwar zwei zweite zweiten zweiter zweites zwischen
""".split())


EN = set("""
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at

back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by

call can cannot ca could

did do does doing done down due during

each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except

few fifteen fifty first five for former formerly forty four from front full
further

get give go

had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred

i if in indeed into is it its itself

keep

last latter latterly least less

just

made make many may me meanwhile might mine more moreover most mostly move much
must my myself

name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere

of off often on once one only onto or other others otherwise our ours ourselves
out over own

part per perhaps please put

quite

rather re really regarding

same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such

take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two

under until up unless upon us used using

various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would

yet you your yours yourself yourselves
""".split())


ES = set("""
actualmente acuerdo adelante ademas además adrede afirmó agregó ahi ahora ahí
al algo alguna algunas alguno algunos algún alli allí alrededor ambos ampleamos
antano antaño ante anterior antes apenas aproximadamente aquel aquella aquellas
aquello aquellos aqui aquél aquélla aquéllas aquéllos aquí arriba arribaabajo
aseguró asi así atras aun aunque ayer añadió aún

bajo bastante bien breve buen buena buenas bueno buenos

cada casi cerca cierta ciertas cierto ciertos cinco claro comentó como con
conmigo conocer conseguimos conseguir considera consideró consigo consigue
consiguen consigues contigo contra cosas creo cual cuales cualquier cuando
cuanta cuantas cuanto cuantos cuatro cuenta cuál cuáles cuándo cuánta cuántas
cuánto cuántos cómo

da dado dan dar de debajo debe deben debido decir dejó del delante demasiado
demás dentro deprisa desde despacio despues después detras detrás dia dias dice
dicen dicho dieron diferente diferentes dijeron dijo dio donde dos durante día
días dónde

ejemplo el ella ellas ello ellos embargo empleais emplean emplear empleas
empleo en encima encuentra enfrente enseguida entonces entre era eramos eran
eras eres es esa esas ese eso esos esta estaba estaban estado estados estais
estamos estan estar estará estas este esto estos estoy estuvo está están ex
excepto existe existen explicó expresó él ésa ésas ése ésos ésta éstas éste
éstos

fin final fue fuera fueron fui fuimos

general gran grandes gueno

ha haber habia habla hablan habrá había habían hace haceis hacemos hacen hacer
hacerlo haces hacia haciendo hago han hasta hay haya he hecho hemos hicieron
hizo horas hoy hubo

igual incluso indicó informo informó intenta intentais intentamos intentan
intentar intentas intento ir

junto

la lado largo las le lejos les llegó lleva llevar lo los luego lugar

mal manera manifestó mas mayor me mediante medio mejor mencionó menos menudo mi
mia mias mientras mio mios mis misma mismas mismo mismos modo momento mucha
muchas mucho muchos muy más mí mía mías mío míos

nada nadie ni ninguna ningunas ninguno ningunos ningún no nos nosotras nosotros
nuestra nuestras nuestro nuestros nueva nuevas nuevo nuevos nunca

ocho os otra otras otro otros

pais para parece parte partir pasada pasado paìs peor pero pesar poca pocas
poco pocos podeis podemos poder podria podriais podriamos podrian podrias podrá
podrán podría podrían poner por porque posible primer primera primero primeros
principalmente pronto propia propias propio propios proximo próximo próximos
pudo pueda puede pueden puedo pues

qeu que quedó queremos quien quienes quiere quiza quizas quizá quizás quién quiénes qué

raras realizado realizar realizó repente respecto

sabe sabeis sabemos saben saber sabes salvo se sea sean segun segunda segundo
según seis ser sera será serán sería señaló si sido siempre siendo siete sigue
siguiente sin sino sobre sois sola solamente solas solo solos somos son soy
soyos su supuesto sus suya suyas suyo sé sí sólo

tal tambien también tampoco tan tanto tarde te temprano tendrá tendrán teneis
tenemos tener tenga tengo tenido tenía tercera ti tiempo tiene tienen toda
todas todavia todavía todo todos total trabaja trabajais trabajamos trabajan
trabajar trabajas trabajo tras trata través tres tu tus tuvo tuya tuyas tuyo
tuyos tú

ultimo un una unas uno unos usa usais usamos usan usar usas uso usted ustedes
última últimas último últimos

va vais valor vamos van varias varios vaya veces ver verdad verdadera verdadero
vez vosotras vosotros voy vuestra vuestras vuestro vuestros

ya yo
""".split())

FI = set("""

aiemmin aika aikaa aikaan aikaisemmin aikaisin aikana aikoina aikoo aikovat
aina ainakaan ainakin ainoa ainoat aiomme aion aiotte aivan ajan alas alemmas
alkuisin alkuun alla alle aloitamme aloitan aloitat aloitatte aloitattivat
aloitettava aloitettavaksi aloitettu aloitimme aloitin aloitit aloititte
aloittaa aloittamatta aloitti aloittivat alta aluksi alussa alusta annettavaksi
annettava annettu ansiosta antaa antamatta antoi apu asia asiaa asian asiasta
asiat asioiden asioihin asioita asti avuksi avulla avun avutta

edelle edelleen edellä edeltä edemmäs edes edessä edestä ehkä ei eikä eilen
eivät eli ellei elleivät ellemme ellen ellet ellette emme en enemmän eniten
ennen ensi ensimmäinen ensimmäiseksi ensimmäisen ensimmäisenä ensimmäiset
ensimmäisiksi ensimmäisinä ensimmäisiä ensimmäistä ensin entinen entisen
entisiä entisten entistä enää eri erittäin erityisesti eräiden eräs eräät esi
esiin esillä esimerkiksi et eteen etenkin ette ettei että

halua haluaa haluamatta haluamme haluan haluat haluatte haluavat halunnut
halusi halusimme halusin halusit halusitte halusivat halutessa haluton he hei
heidän heidät heihin heille heillä heiltä heissä heistä heitä helposti heti
hetkellä hieman hitaasti huolimatta huomenna hyvien hyviin hyviksi hyville
hyviltä hyvin hyvinä hyvissä hyvistä hyviä hyvä hyvät hyvää hän häneen hänelle
hänellä häneltä hänen hänessä hänestä hänet häntä

ihan ilman ilmeisesti itse itsensä itseään

ja jo johon joiden joihin joiksi joilla joille joilta joina joissa joista joita
joka jokainen jokin joko joksi joku jolla jolle jolloin jolta jompikumpi jona
jonka jonkin jonne joo jopa jos joskus jossa josta jota jotain joten jotenkin
jotenkuten jotka jotta jouduimme jouduin jouduit jouduitte joudumme joudun
joudutte joukkoon joukossa joukosta joutua joutui joutuivat joutumaan joutuu
joutuvat juuri jälkeen jälleen jää

kahdeksan kahdeksannen kahdella kahdelle kahdelta kahden kahdessa kahdesta
kahta kahteen kai kaiken kaikille kaikilta kaikkea kaikki kaikkia kaikkiaan
kaikkialla kaikkialle kaikkialta kaikkien kaikkiin kaksi kannalta kannattaa
kanssa kanssaan kanssamme kanssani kanssanne kanssasi kauan kauemmas kaukana
kautta kehen keiden keihin keiksi keille keillä keiltä keinä keissä keistä
keitten keittä keitä keneen keneksi kenelle kenellä keneltä kenen kenenä
kenessä kenestä kenet kenettä kenties kerran kerta kertaa keskellä kesken
keskimäärin ketkä ketä kiitos kohti koko kokonaan kolmas kolme kolmen kolmesti
koska koskaan kovin kuin kuinka kuinkaan kuitenkaan kuitenkin kuka kukaan kukin
kumpainen kumpainenkaan kumpi kumpikaan kumpikin kun kuten kuuden kuusi kuutta
kylliksi kyllä kymmenen kyse

liian liki lisäksi lisää lla luo luona lähekkäin lähelle lähellä läheltä
lähemmäs lähes lähinnä lähtien läpi

mahdollisimman mahdollista me meidän meidät meihin meille meillä meiltä meissä
meistä meitä melkein melko menee menemme menen menet menette menevät meni
menimme menin menit menivät mennessä mennyt menossa mihin miksi mikä mikäli
mikään mille milloin milloinkan millä miltä minkä minne minua minulla minulle
minulta minun minussa minusta minut minuun minä missä mistä miten mitkä mitä
mitään moi molemmat mones monesti monet moni moniaalla moniaalle moniaalta
monta muassa muiden muita muka mukaan mukaansa mukana mutta muu muualla muualle
muualta muuanne muulloin muun muut muuta muutama muutaman muuten myöhemmin myös
myöskin myöskään myötä

ne neljä neljän neljää niiden niihin niiksi niille niillä niiltä niin niinä
niissä niistä niitä noiden noihin noiksi noilla noille noilta noin noina noissa
noista noita nopeammin nopeasti nopeiten nro nuo nyt näiden näihin näiksi
näille näillä näiltä näin näinä näissä näistä näitä nämä

ohi oikea oikealla oikein ole olemme olen olet olette oleva olevan olevat oli
olimme olin olisi olisimme olisin olisit olisitte olisivat olit olitte olivat
olla olleet ollut oma omaa omaan omaksi omalle omalta oman omassa omat omia
omien omiin omiksi omille omilta omissa omista on onkin onko ovat

paikoittain paitsi pakosti paljon paremmin parempi parhaillaan parhaiten
perusteella peräti pian pieneen pieneksi pienelle pienellä pieneltä pienempi
pienestä pieni pienin poikki puolesta puolestaan päälle

runsaasti

saakka sama samaa samaan samalla saman samat samoin sata sataa satojen se
seitsemän sekä sen seuraavat siellä sieltä siihen siinä siis siitä sijaan siksi
sille silloin sillä silti siltä sinne sinua sinulla sinulle sinulta sinun
sinussa sinusta sinut sinuun sinä sisäkkäin sisällä siten sitten sitä ssa sta
suoraan suuntaan suuren suuret suuri suuria suurin suurten

taa taas taemmas tahansa tai takaa takaisin takana takia tallä tapauksessa
tarpeeksi tavalla tavoitteena te teidän teidät teihin teille teillä teiltä
teissä teistä teitä tietysti todella toinen toisaalla toisaalle toisaalta
toiseen toiseksi toisella toiselle toiselta toisemme toisen toisensa toisessa
toisesta toista toistaiseksi toki tosin tuhannen tuhat tule tulee tulemme tulen
tulet tulette tulevat tulimme tulin tulisi tulisimme tulisin tulisit tulisitte
tulisivat tulit tulitte tulivat tulla tulleet tullut tuntuu tuo tuohon tuoksi
tuolla tuolle tuolloin tuolta tuon tuona tuonne tuossa tuosta tuota tuskin tykö
tähän täksi tälle tällä tällöin tältä tämä tämän tänne tänä tänään tässä tästä
täten tätä täysin täytyvät täytyy täällä täältä

ulkopuolella usea useasti useimmiten usein useita uudeksi uudelleen uuden uudet
uusi uusia uusien uusinta uuteen uutta

vaan vai vaiheessa vaikea vaikean vaikeat vaikeilla vaikeille vaikeilta
vaikeissa vaikeista vaikka vain varmasti varsin varsinkin varten vasen
vasemmalla vasta vastaan vastakkain vastan verran vielä vierekkäin vieressä
vieri viiden viime viimeinen viimeisen viimeksi viisi voi voidaan voimme voin
voisi voit voitte voivat vuoden vuoksi vuosi vuosien vuosina vuotta vähemmän
vähintään vähiten vähän välillä

yhdeksän yhden yhdessä yhteen yhteensä yhteydessä yhteyteen yhtä yhtäälle
yhtäällä yhtäältä yhtään yhä yksi yksin yksittäin yleensä ylemmäs yli ylös
ympäri

älköön älä

""".split())


FR = set("""
a à â abord absolument afin ah ai aie ailleurs ainsi ait allaient allo allons
allô alors anterieur anterieure anterieures apres après as assez attendu au
aucun aucune aujourd aujourd'hui aupres auquel aura auraient aurait auront
aussi autre autrefois autrement autres autrui aux auxquelles auxquels avaient
avais avait avant avec avoir avons ayant

bah bas basee bat beau beaucoup bien bigre boum bravo brrr

ça car ce ceci cela celle celle-ci celle-là celles celles-ci celles-là celui
celui-ci celui-là cent cependant certain certaine certaines certains certes ces
cet cette ceux ceux-ci ceux-là chacun chacune chaque cher chers chez chiche
chut chère chères ci cinq cinquantaine cinquante cinquantième cinquième clac
clic combien comme comment comparable comparables compris concernant contre
couic crac

da dans de debout dedans dehors deja delà depuis dernier derniere derriere
derrière des desormais desquelles desquels dessous dessus deux deuxième
deuxièmement devant devers devra different differentes differents différent
différente différentes différents dire directe directement dit dite dits divers
diverse diverses dix dix-huit dix-neuf dix-sept dixième doit doivent donc dont
douze douzième dring du duquel durant dès désormais

effet egale egalement egales eh elle elle-même elles elles-mêmes en encore
enfin entre envers environ es ès est et etaient étaient etais étais etait était
etant étant etc été etre être eu euh eux eux-mêmes exactement excepté extenso
exterieur

fais faisaient faisant fait façon feront fi flac floc font

gens

ha hein hem hep hi ho holà hop hormis hors hou houp hue hui huit huitième hum
hurrah hé hélas i il ils importe

je jusqu jusque juste

la laisser laquelle las le lequel les lesquelles lesquels leur leurs longtemps
lors lorsque lui lui-meme lui-même là lès

ma maint maintenant mais malgre malgré maximale me meme memes merci mes mien
mienne miennes miens mille mince minimale moi moi-meme moi-même moindres moins
mon moyennant multiple multiples même mêmes

na naturel naturelle naturelles ne neanmoins necessaire necessairement neuf
neuvième ni nombreuses nombreux non nos notamment notre nous nous-mêmes nouveau
nul néanmoins nôtre nôtres

o ô oh ohé ollé olé on ont onze onzième ore ou ouf ouias oust ouste outre
ouvert ouverte ouverts où

paf pan par parce parfois parle parlent parler parmi parseme partant
particulier particulière particulièrement pas passé pendant pense permet
personne peu peut peuvent peux pff pfft pfut pif pire plein plouf plus
plusieurs plutôt possessif possessifs possible possibles pouah pour pourquoi
pourrais pourrait pouvait prealable precisement premier première premièrement
pres probable probante procedant proche près psitt pu puis puisque pur pure

qu quand quant quant-à-soi quanta quarante quatorze quatre quatre-vingt
quatrième quatrièmement que quel quelconque quelle quelles quelqu'un quelque
quelques quels qui quiconque quinze quoi quoique

rare rarement rares relative relativement remarquable rend rendre restant reste
restent restrictif retour revoici revoilà rien

sa sacrebleu sait sans sapristi sauf se sein seize selon semblable semblaient
semble semblent sent sept septième sera seraient serait seront ses seul seule
seulement si sien sienne siennes siens sinon six sixième soi soi-même soit
soixante son sont sous souvent specifique specifiques speculatif stop
strictement subtiles suffisant suffisante suffit suis suit suivant suivante
suivantes suivants suivre superpose sur surtout

ta tac tant tardive te tel telle tellement telles tels tenant tend tenir tente
tes tic tien tienne tiennes tiens toc toi toi-même ton touchant toujours tous
tout toute toutefois toutes treize trente tres trois troisième troisièmement
trop très tsoin tsouin tu té

un une unes uniformement unique uniques uns

va vais vas vers via vif vifs vingt vivat vive vives vlan voici voilà vont vos
votre vous vous-mêmes vu vé vôtre vôtres

zut
""".split())


GA = set("""
a ach ag agus an aon ar arna as

ba beirt bhúr

caoga ceathair ceathrar chomh chuig chun cois céad cúig cúigear

daichead dar de deich deichniúr den dhá do don dtí dá dár dó

faoi faoin faoina faoinár fara fiche

gach gan go gur

haon hocht

i iad idir in ina ins inár is

le leis lena lenár

mar mo muid mé

na nach naoi naonúr ná ní níor nó nócha

ocht ochtar ochtó os

roimh

sa seacht seachtar seachtó seasca seisear siad sibh sinn sna sé sí

tar thar thú triúr trí trína trínár tríocha tú

um

ár

é éis

í

ó ón óna ónár
""".split())


HE = set("""
אני
את
אתה
אנחנו
אתן
אתם
הם
הן
היא
הוא
שלי
שלו
שלך
שלה
שלנו
שלכם
שלכן
שלהם
שלהן
לי
לו
לה
לנו
לכם
לכן
להם
להן
אותה
אותו
זה
זאת
אלה
אלו
תחת
מתחת
מעל
בין
עם
עד
נגר
על
אל
מול
של
אצל
כמו
אחר
אותו
בלי
לפני
אחרי
מאחורי
עלי
עליו
עליה
עליך
עלינו
עליכם
לעיכן
עליהם
עליהן
כל
כולם
כולן
כך
ככה
כזה
זה
זות
אותי
אותה
אותם
אותך
אותו
אותן
אותנו
ואת
את
אתכם
אתכן
איתי
איתו
איתך
איתה
איתם
איתן
איתנו
איתכם
איתכן
יהיה
תהיה
היתי
היתה
היה
להיות
עצמי
עצמו
עצמה
עצמם
עצמן
עצמנו
עצמהם
עצמהן
מי
מה
איפה
היכן
במקום שבו
אם
לאן
למקום שבו
מקום בו
איזה
מהיכן
איך
כיצד
באיזו מידה
מתי
בשעה ש
כאשר
כש
למרות
לפני
אחרי
מאיזו סיבה
הסיבה שבגללה
למה
מדוע
לאיזו תכלית
כי
יש
אין
אך
מנין
מאין
מאיפה
יכל
יכלה
יכלו
יכול
יכולה
יכולים
יכולות
יוכלו
יוכל
מסוגל
לא
רק
אולי
אין
לאו
אי
כלל
נגד
אם
עם
אל
אלה
אלו
אף
על
מעל
מתחת
מצד
בשביל
לבין
באמצע
בתוך
דרך
מבעד
באמצעות
למעלה
למטה
מחוץ
מן
לעבר
מכאן
כאן
הנה
הרי
פה
שם
אך
ברם
שוב
אבל
מבלי
בלי
מלבד
רק
בגלל
מכיוון
עד
אשר
ואילו
למרות
אס
כמו
כפי
אז
אחרי
כן
לכן
לפיכך
מאד
עז
מעט
מעטים
במידה
שוב
יותר
מדי
גם
כן
נו
אחר
אחרת
אחרים
אחרות
אשר
או
""".split())


HI = set("""
अंदर
अत
अदि
अप
अपना
अपनि
अपनी
अपने
अभि
अभी
अंदर
आदि
आप
इंहिं
इंहें
इंहों
इतयादि
इत्यादि
इन
इनका
इन्हीं
इन्हें
इन्हों
इस
इसका
इसकि
इसकी
इसके
इसमें
इसि
इसी
इसे
उंहिं
उंहें
उंहों
उन
उनका
उनकि
उनकी
उनके
उनको
उन्हीं
उन्हें
उन्हों
उस
उसके
उसि
उसी
उसे
एक
एवं
एस
एसे
ऐसे
ओर
और
कइ
कई
कर
करता
करते
करना
करने
करें
कहते
कहा
का
काफि
काफ़ी
कि
किंहें
किंहों
कितना
किन्हें
किन्हों
किया
किर
किस
किसि
किसी
किसे
की
कुछ
कुल
के
को
कोइ
कोई
कोन
कोनसा
कौन
कौनसा
गया
घर
जब
जहाँ
जहां
जा
जिंहें
जिंहों
जितना
जिधर
जिन
जिन्हें
जिन्हों
जिस
जिसे
जीधर
जेसा
जेसे
जैसा
जैसे
जो
तक
तब
तरह
तिंहें
तिंहों
तिन
तिन्हें
तिन्हों
तिस
तिसे
तो
था
थि
थी
थे
दबारा
दवारा
दिया
दुसरा
दुसरे
दूसरे
दो
द्वारा
न
नहिं
नहीं
ना
निचे
निहायत
नीचे
ने
पर
पहले
पुरा
पूरा
पे
फिर
बनि
बनी
बहि
बही
बहुत
बाद
बाला
बिलकुल
भि
भितर
भी
भीतर
मगर
मानो
मे
में
यदि
यह
यहाँ
यहां
यहि
यही
या
यिह
ये
रखें
रवासा
रहा
रहे
ऱ्वासा
लिए
लिये
लेकिन
व
वगेरह
वग़ैरह
वरग
वर्ग
वह
वहाँ
वहां
वहिं
वहीं
वाले
वुह
वे
वग़ैरह
संग
सकता
सकते
सबसे
सभि
सभी
साथ
साबुत
साभ
सारा
से
सो
संग
हि
ही
हुअ
हुआ
हुइ
हुई
हुए
हे
हें
है
हैं
हो
होता
होति
होती
होते
होना
होने

""".split())


HR = set("""
a
ako
ali
bi
bih
bila
bili
bilo
bio
bismo
biste
biti
bumo
da
do
duž
ga
hoće
hoćemo
hoćete
hoćeš
hoću
i
iako
ih
ili
iz
ja
je
jedna
jedne
jedno
jer
jesam
jesi
jesmo
jest
jeste
jesu
jim
joj
još
ju
kada
kako
kao
koja
koje
koji
kojima
koju
kroz
li
me
mene
meni
mi
mimo
moj
moja
moje
mu
na
nad
nakon
nam
nama
nas
naš
naša
naše
našeg
ne
nego
neka
neki
nekog
neku
nema
netko
neće
nećemo
nećete
nećeš
neću
nešto
ni
nije
nikoga
nikoje
nikoju
nisam
nisi
nismo
niste
nisu
njega
njegov
njegova
njegovo
njemu
njezin
njezina
njezino
njih
njihov
njihova
njihovo
njim
njima
njoj
nju
no
o
od
odmah
on
ona
oni
ono
ova
pa
pak
po
pod
pored
prije
s
sa
sam
samo
se
sebe
sebi
si
smo
ste
su
sve
svi
svog
svoj
svoja
svoje
svom
ta
tada
taj
tako
te
tebe
tebi
ti
to
toj
tome
tu
tvoj
tvoja
tvoje
u
uz
vam
vama
vas
vaš
vaša
vaše
već
vi
vrlo
za
zar
će
ćemo
ćete
ćeš
ću
što
""".split())


HU = set("""
a abban ahhoz ahogy ahol aki akik akkor akár alatt amely amelyek amelyekben
amelyeket amelyet amelynek ami amikor amit amolyan amíg annak arra arról az
azok azon azonban azt aztán azután azzal azért

be belül benne bár

cikk cikkek cikkeket csak

de

e ebben eddig egy egyes egyetlen egyik egyre egyéb egész ehhez ekkor el ellen
elo eloször elott elso elég előtt emilyen ennek erre ez ezek ezen ezt ezzel
ezért

fel felé

ha hanem hiszen hogy hogyan hát

ide igen ill ill. illetve ilyen ilyenkor inkább is ismét ison itt

jobban jó jól

kell kellett keressünk keresztül ki kívül között közül

le legalább legyen lehet lehetett lenne lenni lesz lett

ma maga magát majd meg mellett mely melyek mert mi miatt mikor milyen minden
mindenki mindent mindig mint mintha mit mivel miért mondta most már más másik
még míg

nagy nagyobb nagyon ne nekem neki nem nincs néha néhány nélkül

o oda ok oket olyan ott

pedig persze például

rá

s saját sem semmi sok sokat sokkal stb. szemben szerint szinte számára szét

talán te tehát teljes ti tovább továbbá több túl ugyanis

utolsó után utána

vagy vagyis vagyok valaki valami valamint való van vannak vele vissza viszont
volna volt voltak voltam voltunk

által általában át

én éppen és

így

ön össze

úgy új újabb újra

ő őket
""".split())


ID = set("""
ada
adalah
adanya
adapun
agak
agaknya
agar
akan
akankah
akhir
akhiri
akhirnya
aku
akulah
amat
amatlah
anda
andalah
antar
antara
antaranya
apa
apaan
apabila
apakah
apalagi
apatah
artinya
asal
asalkan
atas
atau
ataukah
ataupun
awal
awalnya
bagai
bagaikan
bagaimana
bagaimanakah
bagaimanapun
bagi
bagian
bahkan
bahwa
bahwasanya
baik
bakal
bakalan
balik
banyak
bapak
baru
bawah
beberapa
begini
beginian
beginikah
beginilah
begitu
begitukah
begitulah
begitupun
bekerja
belakang
belakangan
belum
belumlah
benar
benarkah
benarlah
berada
berakhir
berakhirlah
berakhirnya
berapa
berapakah
berapalah
berapapun
berarti
berawal
berbagai
berdatangan
beri
berikan
berikut
berikutnya
berjumlah
berkali-kali
berkata
berkehendak
berkeinginan
berkenaan
berlainan
berlalu
berlangsung
berlebihan
bermacam
bermacam-macam
bermaksud
bermula
bersama
bersama-sama
bersiap
bersiap-siap
bertanya
bertanya-tanya
berturut
berturut-turut
bertutur
berujar
berupa
besar
betul
betulkah
biasa
biasanya
bila
bilakah
bisa
bisakah
boleh
bolehkah
bolehlah
buat
bukan
bukankah
bukanlah
bukannya
bulan
bung
cara
caranya
cukup
cukupkah
cukuplah
cuma
dahulu
dalam
dan
dapat
dari
daripada
datang
dekat
demi
demikian
demikianlah
dengan
depan
di
dia
diakhiri
diakhirinya
dialah
diantara
diantaranya
diberi
diberikan
diberikannya
dibuat
dibuatnya
didapat
didatangkan
digunakan
diibaratkan
diibaratkannya
diingat
diingatkan
diinginkan
dijawab
dijelaskan
dijelaskannya
dikarenakan
dikatakan
dikatakannya
dikerjakan
diketahui
diketahuinya
dikira
dilakukan
dilalui
dilihat
dimaksud
dimaksudkan
dimaksudkannya
dimaksudnya
diminta
dimintai
dimisalkan
dimulai
dimulailah
dimulainya
dimungkinkan
dini
dipastikan
diperbuat
diperbuatnya
dipergunakan
diperkirakan
diperlihatkan
diperlukan
diperlukannya
dipersoalkan
dipertanyakan
dipunyai
diri
dirinya
disampaikan
disebut
disebutkan
disebutkannya
disini
disinilah
ditambahkan
ditandaskan
ditanya
ditanyai
ditanyakan
ditegaskan
ditujukan
ditunjuk
ditunjuki
ditunjukkan
ditunjukkannya
ditunjuknya
dituturkan
dituturkannya
diucapkan
diucapkannya
diungkapkan
dong
dua
dulu
empat
enggak
enggaknya
entah
entahlah
guna
gunakan
hal
hampir
hanya
hanyalah
hari
harus
haruslah
harusnya
hendak
hendaklah
hendaknya
hingga
ia
ialah
ibarat
ibaratkan
ibaratnya
ibu
ikut
ingat
ingat-ingat
ingin
inginkah
inginkan
ini
inikah
inilah
itu
itukah
itulah
jadi
jadilah
jadinya
jangan
jangankan
janganlah
jauh
jawab
jawaban
jawabnya
jelas
jelaskan
jelaslah
jelasnya
jika
jikalau
juga
jumlah
jumlahnya
justru
kala
kalau
kalaulah
kalaupun
kalian
kami
kamilah
kamu
kamulah
kan
kapan
kapankah
kapanpun
karena
karenanya
kasus
kata
katakan
katakanlah
katanya
ke
keadaan
kebetulan
kecil
kedua
keduanya
keinginan
kelamaan
kelihatan
kelihatannya
kelima
keluar
kembali
kemudian
kemungkinan
kemungkinannya
kenapa
kepada
kepadanya
kesampaian
keseluruhan
keseluruhannya
keterlaluan
ketika
khususnya
kini
kinilah
kira
kira-kira
kiranya
kita
kitalah
kok
kurang
lagi
lagian
lah
lain
lainnya
lalu
lama
lamanya
lanjut
lanjutnya
lebih
lewat
lima
luar
macam
maka
makanya
makin
malah
malahan
mampu
mampukah
mana
manakala
manalagi
masa
masalah
masalahnya
masih
masihkah
masing
masing-masing
mau
maupun
melainkan
melakukan
melalui
melihat
melihatnya
memang
memastikan
memberi
memberikan
membuat
memerlukan
memihak
meminta
memintakan
memisalkan
memperbuat
mempergunakan
memperkirakan
memperlihatkan
mempersiapkan
mempersoalkan
mempertanyakan
mempunyai
memulai
memungkinkan
menaiki
menambahkan
menandaskan
menanti
menanti-nanti
menantikan
menanya
menanyai
menanyakan
mendapat
mendapatkan
mendatang
mendatangi
mendatangkan
menegaskan
mengakhiri
mengapa
mengatakan
mengatakannya
mengenai
mengerjakan
mengetahui
menggunakan
menghendaki
mengibaratkan
mengibaratkannya
mengingat
mengingatkan
menginginkan
mengira
mengucapkan
mengucapkannya
mengungkapkan
menjadi
menjawab
menjelaskan
menuju
menunjuk
menunjuki
menunjukkan
menunjuknya
menurut
menuturkan
menyampaikan
menyangkut
menyatakan
menyebutkan
menyeluruh
menyiapkan
merasa
mereka
merekalah
merupakan
meski
meskipun
meyakini
meyakinkan
minta
mirip
misal
misalkan
misalnya
mula
mulai
mulailah
mulanya
mungkin
mungkinkah
nah
naik
namun
nanti
nantinya
nyaris
nyatanya
oleh
olehnya
pada
padahal
padanya
pak
paling
panjang
pantas
para
pasti
pastilah
penting
pentingnya
per
percuma
perlu
perlukah
perlunya
pernah
persoalan
pertama
pertama-tama
pertanyaan
pertanyakan
pihak
pihaknya
pukul
pula
pun
punya
rasa
rasanya
rata
rupanya
saat
saatnya
saja
sajalah
saling
sama
sama-sama
sambil
sampai
sampai-sampai
sampaikan
sana
sangat
sangatlah
satu
saya
sayalah
se
sebab
sebabnya
sebagai
sebagaimana
sebagainya
sebagian
sebaik
sebaik-baiknya
sebaiknya
sebaliknya
sebanyak
sebegini
sebegitu
sebelum
sebelumnya
sebenarnya
seberapa
sebesar
sebetulnya
sebisanya
sebuah
sebut
sebutlah
sebutnya
secara
secukupnya
sedang
sedangkan
sedemikian
sedikit
sedikitnya
seenaknya
segala
segalanya
segera
seharusnya
sehingga
seingat
sejak
sejauh
sejenak
sejumlah
sekadar
sekadarnya
sekali
sekali-kali
sekalian
sekaligus
sekalipun
sekarang
sekarang
sekecil
seketika
sekiranya
sekitar
sekitarnya
sekurang-kurangnya
sekurangnya
sela
selain
selaku
selalu
selama
selama-lamanya
selamanya
selanjutnya
seluruh
seluruhnya
semacam
semakin
semampu
semampunya
semasa
semasih
semata
semata-mata
semaunya
sementara
semisal
semisalnya
sempat
semua
semuanya
semula
sendiri
sendirian
sendirinya
seolah
seolah-olah
seorang
sepanjang
sepantasnya
sepantasnyalah
seperlunya
seperti
sepertinya
sepihak
sering
seringnya
serta
serupa
sesaat
sesama
sesampai
sesegera
sesekali
seseorang
sesuatu
sesuatunya
sesudah
sesudahnya
setelah
setempat
setengah
seterusnya
setiap
setiba
setibanya
setidak-tidaknya
setidaknya
setinggi
seusai
sewaktu
siap
siapa
siapakah
siapapun
sini
sinilah
soal
soalnya
suatu
sudah
sudahkah
sudahlah
supaya
tadi
tadinya
tahu
tahun
tak
tambah
tambahnya
tampak
tampaknya
tandas
tandasnya
tanpa
tanya
tanyakan
tanyanya
tapi
tegas
tegasnya
telah
tempat
tengah
tentang
tentu
tentulah
tentunya
tepat
terakhir
terasa
terbanyak
terdahulu
terdapat
terdiri
terhadap
terhadapnya
teringat
teringat-ingat
terjadi
terjadilah
terjadinya
terkira
terlalu
terlebih
terlihat
termasuk
ternyata
tersampaikan
tersebut
tersebutlah
tertentu
tertuju
terus
terutama
tetap
tetapi
tiap
tiba
tiba-tiba
tidak
tidakkah
tidaklah
tiga
tinggi
toh
tunjuk
turut
tutur
tuturnya
ucap
ucapnya
ujar
ujarnya
umum
umumnya
ungkap
ungkapnya
untuk
usah
usai
waduh
wah
wahai
waktu
waktunya
walau
walaupun
wong
yaitu
yakin
yakni
yang
""".split())


IT = set("""
a abbastanza abbia abbiamo abbiano abbiate accidenti ad adesso affinche agl
agli ahime ahimè ai al alcuna alcuni alcuno all alla alle allo allora altri
altrimenti altro altrove altrui anche ancora anni anno ansa anticipo assai
attesa attraverso avanti avemmo avendo avente aver avere averlo avesse
avessero avessi avessimo aveste avesti avete aveva avevamo avevano avevate
avevi avevo avrai avranno avrebbe avrebbero avrei avremmo avremo avreste
avresti avrete avrà avrò avuta avute avuti avuto

basta bene benissimo brava bravo

casa caso cento certa certe certi certo che chi chicchessia chiunque ci
ciascuna ciascuno cima cio cioe circa citta città co codesta codesti codesto
cogli coi col colei coll coloro colui come cominci comunque con concernente
conciliarsi conclusione consiglio contro cortesia cos cosa cosi così cui

da dagl dagli dai dal dall dalla dalle dallo dappertutto davanti degl degli
dei del dell della delle dello dentro detto deve di dice dietro dire
dirimpetto diventa diventare diventato dopo dov dove dovra dovrà dovunque due
dunque durante

ebbe ebbero ebbi ecc ecco ed effettivamente egli ella entrambi eppure era
erano eravamo eravate eri ero esempio esse essendo esser essere essi ex

fa faccia facciamo facciano facciate faccio facemmo facendo facesse facessero
facessi facessimo faceste facesti faceva facevamo facevano facevate facevi
facevo fai fanno farai faranno fare farebbe farebbero farei faremmo faremo
fareste faresti farete farà farò fatto favore fece fecero feci fin finalmente
finche fine fino forse forza fosse fossero fossi fossimo foste fosti fra
frattempo fu fui fummo fuori furono futuro generale

gia già giacche giorni giorno gli gliela gliele glieli glielo gliene governo
grande grazie gruppo

ha haha hai hanno ho

ieri il improvviso in inc infatti inoltre insieme intanto intorno invece io

la là lasciato lato lavoro le lei li lo lontano loro lui lungo luogo

ma macche magari maggior mai male malgrado malissimo mancanza marche me
medesimo mediante meglio meno mentre mesi mezzo mi mia mie miei mila miliardi
milioni minimi ministro mio modo molti moltissimo molto momento mondo mosto

nazionale ne negl negli nei nel nell nella nelle nello nemmeno neppure nessun
nessuna nessuno niente no noi non nondimeno nonostante nonsia nostra nostre
nostri nostro novanta nove nulla nuovo

od oggi ogni ognuna ognuno oltre oppure ora ore osi ossia ottanta otto

paese parecchi parecchie parecchio parte partendo peccato peggio per perche
perché percio perciò perfino pero persino persone però piedi pieno piglia piu
piuttosto più po pochissimo poco poi poiche possa possedere posteriore posto
potrebbe preferibilmente presa press prima primo principalmente probabilmente
proprio puo può pure purtroppo

qualche qualcosa qualcuna qualcuno quale quali qualunque quando quanta quante
quanti quanto quantunque quasi quattro quel quella quelle quelli quello quest
questa queste questi questo qui quindi

realmente recente recentemente registrazione relativo riecco salvo

sara sarà sarai saranno sarebbe sarebbero sarei saremmo saremo sareste
saresti sarete saro sarò scola scopo scorso se secondo seguente seguito sei
sembra sembrare sembrato sembri sempre senza sette si sia siamo siano siate
siete sig solito solo soltanto sono sopra sotto spesso srl sta stai stando
stanno starai staranno starebbe starebbero starei staremmo staremo stareste
staresti starete starà starò stata state stati stato stava stavamo stavano
stavate stavi stavo stemmo stessa stesse stessero stessi stessimo stesso
steste stesti stette stettero stetti stia stiamo stiano stiate sto su sua
subito successivamente successivo sue sugl sugli sui sul sull sulla sulle
sullo suo suoi

tale tali talvolta tanto te tempo ti titolo torino tra tranne tre trenta
troppo trovato tu tua tue tuo tuoi tutta tuttavia tutte tutti tutto

uguali ulteriore ultimo un una uno uomo

va vale vari varia varie vario verso vi via vicino visto vita voi volta volte
vostra vostre vostri vostro
""".split())


NB = set("""
alle allerede alt and andre annen annet at av

bak bare bedre beste blant ble bli blir blitt bris by både

da dag de del dem den denne der dermed det dette disse drept du

eller en enn er et ett etter

fem fikk fire fjor flere folk for fortsatt fotball fra fram frankrike fredag
funnet få får fått før først første

gang gi gikk gjennom gjorde gjort gjør gjøre god godt grunn gå går

ha hadde ham han hans har hele helt henne hennes her hun hva hvor hvordan
hvorfor

i ifølge igjen ikke ingen inn

ja jeg

kamp kampen kan kl klart kom komme kommer kontakt kort kroner kunne kveld
kvinner

la laget land landet langt leder ligger like litt løpet lørdag

man mandag mange mannen mars med meg mellom men mener menn mennesker mens mer
millioner minutter mot msci mye må mål måtte

ned neste noe noen nok norge norsk norske ntb ny nye nå når

og også om onsdag opp opplyser oslo oss over

personer plass poeng politidistrikt politiet president prosent på

regjeringen runde rundt russland

sa saken samme sammen samtidig satt se seg seks selv senere september ser sett
siden sier sin sine siste sitt skal skriver skulle slik som sted stedet stor
store står sverige svært så søndag

ta tatt tid tidligere til tilbake tillegg tirsdag to tok torsdag tre tror
tyskland

under usa ut uten utenfor

vant var ved veldig vi videre viktig vil ville viser vår være vært

å år

ønsker
""".split())


NL = set("""
aan af al alles als altijd andere

ben bij

daar dan dat de der deze die dit doch doen door dus

een eens en er

ge geen geweest

haar had heb hebben heeft hem het hier hij hoe hun

iemand iets ik in is

ja je

kan kon kunnen

maar me meer men met mij mijn moet

na naar niet niets nog nu

of om omdat ons ook op over

reeds

te tegen toch toen tot

u uit uw

van veel voor

want waren was wat we wel werd wezen wie wij wil worden

zal ze zei zelf zich zij zijn zo zonder zou
""".split())


PL = set("""
ach aj albo

bardzo bez bo być

ci cię ciebie co czy

daleko dla dlaczego dlatego do dobrze dokąd dość dużo dwa dwaj dwie dwoje dziś
dzisiaj

gdyby gdzie

go

ich ile im inny

ja ją jak jakby jaki je jeden jedna jedno jego jej jemu jeśli jest jestem
jeżeli już

każdy kiedy kierunku kto ku

lub

ma mają mam mi mną mnie moi mój moja moje może mu my

na nam nami nas nasi nasz nasza nasze natychmiast nią nic nich nie niego niej
niemu nigdy nim nimi niż

obok od około on ona one oni ono owszem

po pod ponieważ przed przedtem

są sam sama się skąd

tak taki tam ten to tobą tobie tu tutaj twoi twój twoja twoje ty

wam wami was wasi wasz wasza wasze we więc wszystko wtedy wy

żaden zawsze że
""".split())


PT = set("""
à às acerca adeus agora ainda algo algumas alguns ali além ambas ambos ano
anos antes ao aos apenas apoio apoia apontar após aquela aquelas aquele aqueles
aqui aquilo área as assim através atrás até aí

baixo bastante bem boa bom breve

cada caminho catorze cedo cento certamente certeza cima cinco coisa com como
comprido comprida conhecida conhecido conselho contra corrente custa cá

da daquela daquele dar das de debaixo demais dentro depois desde desligada
desligado dessa desse desta deste deve devem deverá dez dezanove dezasseis
dezassete dezoito dia diante direita diz dizem dizer do dois dos doze duas dá
dão dúvida

é ela elas ele eles em embora enquanto entre então era és essa essas esse esses
esta estado estar estará estas estava este estes esteve estive estivemos
estiveram estiveste estivestes estou está estás estão eu exemplo

falta fará favor faz fazeis fazem fazemos fazer fazes fazia faço fez fim final
foi fomos for fora foram forma foste fostes fui

geral grande grandes grupo

hoje horas há

iniciar inicio ir irá isso isto já

lado ligado local logo longe lugar lá

maior maioria maiorias mais mal mas me meio menor menos meses mesmo meu meus
mil minha minhas momento muito muitos máximo mês

na nada naquela naquele nas nem nenhuma nessa nesse nesta neste no noite nome
nos nossa nossas nosso nossos nova novas nove novo novos num numa nunca nuns
não nível nós número números

obra obrigada obrigado oitava oitavo oito onde ontem onze os ou outra outras
outro outros

para parece parte partir pegar pela pelas pelo pelos perto pessoas pode podem
poder poderá podia ponto pontos por porque porquê posição possivelmente posso
possível pouca pouco povo primeira primeiro próprio próxima próximo puderam pôde
põe põem

qual qualquer quando quanto quarta quarto quatro que quem quer querem quero
questão quieta quieto quinta quinto quinze quê

relação

sabe saber se segunda segundo sei seis sem sempre ser seria sete seu seus sexta
sexto sim sistema sob sobre sois somente somos sou sua suas são sétima sétimo

tal talvez também tanta tanto tarde te tem temos tempo tendes tenho tens tentar
tentaram tente tentei ter terceira terceiro teu teus teve tipo tive tivemos
tiveram tiveste tivestes toda todas todo todos trabalhar trabalho treze três tu
tua tuas tudo tão têm

último um uma umas uns usa usar

vai vais valor veja vem vens ver verdade verdadeira verdadeiro vez vezes viagem
vinda vindo vinte você vocês vos vossa vossas vosso vossos vários vão vêm vós

zero
""".split())


RO = set("""
a
abia
acea
aceasta
această
aceea
aceeasi
acei
aceia
acel
acela
acelasi
acele
acelea
acest
acesta
aceste
acestea
acestei
acestia
acestui
aceşti
aceştia
acolo
acord
acum
adica
ai
aia
aibă
aici
aiurea
al
ala
alaturi
ale
alea
alt
alta
altceva
altcineva
alte
altfel
alti
altii
altul
am
anume
apoi
ar
are
as
asa
asemenea
asta
astazi
astea
astfel
astăzi
asupra
atare
atat
atata
atatea
atatia
ati
atit
atita
atitea
atitia
atunci
au
avea
avem
aveţi
avut
azi
aş
aşadar
aţi
b
ba
bine
bucur
bună
c
ca
cam
cand
capat
care
careia
carora
caruia
cat
catre
caut
ce
cea
ceea
cei
ceilalti
cel
cele
celor
ceva
chiar
ci
cinci
cind
cine
cineva
cit
cita
cite
citeva
citi
citiva
conform
contra
cu
cui
cum
cumva
curând
curînd
când
cât
câte
câtva
câţi
cînd
cît
cîte
cîtva
cîţi
că
căci
cărei
căror
cărui
către
d
da
daca
dacă
dar
dat
datorită
dată
dau
de
deasupra
deci
decit
degraba
deja
deoarece
departe
desi
despre
deşi
din
dinaintea
dintr
dintr-
dintre
doar
doi
doilea
două
drept
dupa
după
dă
e
ea
ei
el
ele
era
eram
este
eu
exact
eşti
f
face
fara
fata
fel
fi
fie
fiecare
fii
fim
fiu
fiţi
foarte
fost
frumos
fără
g
geaba
graţie
h
halbă
i
ia
iar
ieri
ii
il
imi
in
inainte
inapoi
inca
incit
insa
intr
intre
isi
iti
j
k
l
la
le
li
lor
lui
lângă
lîngă
m
ma
mai
mare
mea
mei
mele
mereu
meu
mi
mie
mine
mod
mult
multa
multe
multi
multă
mulţi
mulţumesc
mâine
mîine
mă
n
ne
nevoie
ni
nici
niciodata
nicăieri
nimeni
nimeri
nimic
niste
nişte
noastre
noastră
noi
noroc
nostri
nostru
nou
noua
nouă
noştri
nu
numai
o
opt
or
ori
oricare
orice
oricine
oricum
oricând
oricât
oricînd
oricît
oriunde
p
pai
parca
patra
patru
patrulea
pe
pentru
peste
pic
pina
plus
poate
pot
prea
prima
primul
prin
printr-
putini
puţin
puţina
puţină
până
pînă
r
rog
s
sa
sa-mi
sa-ti
sai
sale
sau
se
si
sint
sintem
spate
spre
sub
sunt
suntem
sunteţi
sus
sută
sînt
sîntem
sînteţi
să
săi
său
t
ta
tale
te
ti
timp
tine
toata
toate
toată
tocmai
tot
toti
totul
totusi
totuşi
toţi
trei
treia
treilea
tu
tuturor
tăi
tău
u
ul
ului
un
una
unde
undeva
unei
uneia
unele
uneori
unii
unor
unora
unu
unui
unuia
unul
v
va
vi
voastre
voastră
voi
vom
vor
vostru
vouă
voştri
vreme
vreo
vreun
vă
x
z
zece
zero
zi
zice
îi
îl
îmi
împotriva
în
înainte
înaintea
încotro
încât
încît
între
întrucât
întrucît
îţi
ăla
ălea
ăsta
ăstea
ăştia
şapte
şase
şi
ştiu
ţi
ţie
""".split())


RU = set("""
а

будем будет будете будешь буду будут будучи будь будьте бы был была были было
быть

в вам вами вас весь во вот все всё всего всей всем всём всеми всему всех всею
всея всю вся вы

да для до

его едим едят ее её ей ел ела ем ему емъ если ест есть ешь еще ещё ею

же

за

и из или им ими имъ их

к как кем ко когда кого ком кому комья которая которого которое которой котором
которому которою которую которые который которым которыми которых кто

меня мне мной мною мог моги могите могла могли могло могу могут мое моё моего
моей моем моём моему моею можем может можете можешь мои мой моим моими моих
мочь мою моя мы

на нам нами нас наса наш наша наше нашего нашей нашем нашему нашею наши нашим
нашими наших нашу не него нее неё ней нем нём нему нет нею ним ними них но

о об один одна одни одним одними одних одно одного одной одном одному одною
одну он она оне они оно от

по при

с сам сама сами самим самими самих само самого самом самому саму свое своё
своего своей своем своём своему своею свои свой своим своими своих свою своя
себе себя собой собою

та так такая такие таким такими таких такого такое такой таком такому такою
такую те тебе тебя тем теми тех то тобой тобою того той только том томах тому
тот тою ту ты

у уже

чего чем чём чему что чтобы

эта эти этим этими этих это этого этой этом этому этот этою эту

я
""".split())


SV = set("""
aderton adertonde adjö aldrig alla allas allt alltid alltså än andra andras
annan annat ännu artonde arton åtminstone att åtta åttio åttionde åttonde av
även

båda bådas bakom bara bäst bättre behöva behövas behövde behövt beslut beslutat
beslutit bland blev bli blir blivit bort borta bra

då dag dagar dagarna dagen där därför de del delen dem den deras dess det detta
dig din dina dit ditt dock du

efter eftersom elfte eller elva en enkel enkelt enkla enligt er era ert ett
ettusen

få fanns får fått fem femte femtio femtionde femton femtonde fick fin finnas
finns fjärde fjorton fjortonde fler flera flesta följande för före förlåt förra
första fram framför från fyra fyrtio fyrtionde

gå gälla gäller gällt går gärna gått genast genom gick gjorde gjort god goda
godare godast gör göra gott

ha hade haft han hans har här heller hellre helst helt henne hennes hit hög
höger högre högst hon honom hundra hundraen hundraett hur

i ibland idag igår igen imorgon in inför inga ingen ingenting inget innan inne
inom inte inuti

ja jag jämfört

kan kanske knappast kom komma kommer kommit kr kunde kunna kunnat kvar

länge längre långsam långsammare långsammast långsamt längst långt lätt lättare
lättast legat ligga ligger lika likställd likställda lilla lite liten litet

man många måste med mellan men mer mera mest mig min mina mindre minst mitt
mittemot möjlig möjligen möjligt möjligtvis mot mycket

någon någonting något några när nästa ned nederst nedersta nedre nej ner ni nio
nionde nittio nittionde nitton nittonde nödvändig nödvändiga nödvändigt
nödvändigtvis nog noll nr nu nummer

och också ofta oftast olika olikt om oss

över övermorgon överst övre

på

rakt rätt redan

så sade säga säger sagt samma sämre sämst sedan senare senast sent sex sextio
sextionde sexton sextonde sig sin sina sist sista siste sitt sjätte sju sjunde
sjuttio sjuttionde sjutton sjuttonde ska skall skulle slutligen små smått snart
som stor stora större störst stort

tack tidig tidigare tidigast tidigt till tills tillsammans tio tionde tjugo
tjugoen tjugoett tjugonde tjugotre tjugotvå tjungo tolfte tolv tre tredje
trettio trettionde tretton trettonde två tvåhundra

under upp ur ursäkt ut utan utanför ute

vad vänster vänstra var vår vara våra varför varifrån varit varken värre
varsågod vart vårt vem vems verkligen vi vid vidare viktig viktigare viktigast
viktigt vilka vilken vilket vill
""".split())


TR = set("""
acaba
acep
adamakıllı
adeta
ait
altmýþ
altmış
altý
altı
ama
amma
anca
ancak
arada
artýk
aslında
aynen
ayrıca
az
açıkça
açıkçası
bana
bari
bazen
bazý
bazı
başkası
baţka
belki
ben
benden
beni
benim
beri
beriki
beþ
beş
beţ
bilcümle
bile
bin
binaen
binaenaleyh
bir
biraz
birazdan
birbiri
birden
birdenbire
biri
birice
birileri
birisi
birkaç
birkaçı
birkez
birlikte
birçok
birçoğu
birþey
birþeyi
birşey
birşeyi
birţey
bitevi
biteviye
bittabi
biz
bizatihi
bizce
bizcileyin
bizden
bize
bizi
bizim
bizimki
bizzat
boşuna
bu
buna
bunda
bundan
bunlar
bunları
bunların
bunu
bunun
buracıkta
burada
buradan
burası
böyle
böylece
böylecene
böylelikle
böylemesine
böylesine
büsbütün
bütün
cuk
cümlesi
da
daha
dahi
dahil
dahilen
daima
dair
dayanarak
de
defa
dek
demin
demincek
deminden
denli
derakap
derhal
derken
deđil
değil
değin
diye
diđer
diğer
diğeri
doksan
dokuz
dolayı
dolayısıyla
doğru
dört
edecek
eden
ederek
edilecek
ediliyor
edilmesi
ediyor
elbet
elbette
elli
emme
en
enikonu
epey
epeyce
epeyi
esasen
esnasında
etmesi
etraflı
etraflıca
etti
ettiği
ettiğini
evleviyetle
evvel
evvela
evvelce
evvelden
evvelemirde
evveli
eđer
eğer
fakat
filanca
gah
gayet
gayetle
gayri
gayrı
gelgelelim
gene
gerek
gerçi
geçende
geçenlerde
gibi
gibilerden
gibisinden
gine
göre
gırla
hakeza
halbuki
halen
halihazırda
haliyle
handiyse
hangi
hangisi
hani
hariç
hasebiyle
hasılı
hatta
hele
hem
henüz
hep
hepsi
her
herhangi
herkes
herkesin
hiç
hiçbir
hiçbiri
hoş
hulasaten
iken
iki
ila
ile
ilen
ilgili
ilk
illa
illaki
imdi
indinde
inen
insermi
ise
ister
itibaren
itibariyle
itibarıyla
iyi
iyice
iyicene
için
iş
işte
iţte
kadar
kaffesi
kah
kala
kanýmca
karşın
katrilyon
kaynak
kaçı
kelli
kendi
kendilerine
kendini
kendisi
kendisine
kendisini
kere
kez
keza
kezalik
keşke
keţke
ki
kim
kimden
kime
kimi
kimisi
kimse
kimsecik
kimsecikler
külliyen
kýrk
kýsaca
kırk
kısaca
lakin
leh
lütfen
maada
madem
mademki
mamafih
mebni
međer
meğer
meğerki
meğerse
milyar
milyon
mu
mü
mý
mı
nasýl
nasıl
nasılsa
nazaran
naşi
ne
neden
nedeniyle
nedenle
nedense
nerde
nerden
nerdeyse
nere
nerede
nereden
neredeyse
neresi
nereye
netekim
neye
neyi
neyse
nice
nihayet
nihayetinde
nitekim
niye
niçin
o
olan
olarak
oldu
olduklarını
oldukça
olduğu
olduğunu
olmadı
olmadığı
olmak
olması
olmayan
olmaz
olsa
olsun
olup
olur
olursa
oluyor
on
ona
onca
onculayın
onda
ondan
onlar
onlardan
onlari
onlarýn
onları
onların
onu
onun
oracık
oracıkta
orada
oradan
oranca
oranla
oraya
otuz
oysa
oysaki
pek
pekala
peki
pekçe
peyderpey
rağmen
sadece
sahi
sahiden
sana
sanki
sekiz
seksen
sen
senden
seni
senin
siz
sizden
sizi
sizin
sonra
sonradan
sonraları
sonunda
tabii
tam
tamam
tamamen
tamamıyla
tarafından
tek
trilyon
tüm
var
vardı
vasıtasıyla
ve
velev
velhasıl
velhasılıkelam
veya
veyahut
ya
yahut
yakinen
yakında
yakından
yakınlarda
yalnız
yalnızca
yani
yapacak
yapmak
yaptı
yaptıkları
yaptığı
yaptığını
yapılan
yapılması
yapıyor
yedi
yeniden
yenilerde
yerine
yetmiþ
yetmiş
yetmiţ
yine
yirmi
yok
yoksa
yoluyla
yüz
yüzünden
zarfında
zaten
zati
zira
çabuk
çabukça
çeşitli
çok
çokları
çoklarınca
çokluk
çoklukla
çokça
çoğu
çoğun
çoğunca
çoğunlukla
çünkü
öbür
öbürkü
öbürü
önce
önceden
önceleri
öncelikle
öteki
ötekisi
öyle
öylece
öylelikle
öylemesine
öz
üzere
üç
þey
þeyden
þeyi
þeyler
þu
þuna
þunda
þundan
þunu
şayet
şey
şeyden
şeyi
şeyler
şu
şuna
şuncacık
şunda
şundan
şunlar
şunları
şunu
şunun
şura
şuracık
şuracıkta
şurası
şöyle
ţayet
ţimdi
ţu
ţöyle
""".split())


EMAILS = set(['etc','eg','ie','im','dont','doesnt',
              'email','e_mail','e_mailed','e_mails',
              'emails', 'ive','let','ill','tel',
              'attachments','tel','sent','disclaimer',
              'ph','hi','dear','following','link',
              'find','attached','isnt','id'])

STOPS = {
    'bn': BN,
    'da': DA,
    'de': DE,
    'en': EN,
    'em': EMAILS,
    'es': ES,
    'fi': FI,
    'fr': FR,
    'ga': GA,
    'he': HE,
    'hi': HI,
    'hr': HR,
    'hu': HU,
    'id': ID,
    'it': IT,
    'nb': NB,
    'nl': NL,
    'pl': PL,
    'pt': PT,
    'ro': RO,
    'ru': RU,
    'sv': SV,
    'tr': TR
}

def get_stops(langs=['en']):
    stopwords = list()
    for lang_id in langs:
        stopwords += list(STOPS[lang_id])
    return set(stopwords)