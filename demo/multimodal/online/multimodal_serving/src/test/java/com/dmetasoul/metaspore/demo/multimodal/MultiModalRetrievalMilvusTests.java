//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package com.dmetasoul.metaspore.demo.multimodal;

import com.dmetasoul.metaspore.demo.multimodal.service.MilvusService;
import io.milvus.response.SearchResultsWrapper;
import org.assertj.core.util.Lists;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

@SpringBootTest
public class MultiModalRetrievalMilvusTests {

    @Autowired
    private ApplicationContext context;

    @Test
    void contextLoads() {
    }

    @Test
    public void testQueryMilvusByEmbeddingOfBaikeQa() {
        System.out.println("Test query Milvus Service(baike-qa) by embeddings:");
        MilvusService service = context.getBean(MilvusService.class);

        Map<String, String> args = new HashMap<>();
        args.put("collectionName", "baike_qa_demo");
        args.put("outFields", "id");
        args.put("vectorField", "question_emb");
        service.setMilvusArgs(args);

        // List<List<Float>> vectors = generateFloatVectors(3, 768);
        List<List<Float>> vectors = generateFloatVectors();
        Map<Integer, List<SearchResultsWrapper.IDScore>> result = service.findByEmbeddingVectors(vectors, 5);
        System.out.println(result);
    }

    @Test
    public void testQueryMilvusByEmbeddingOfTxtToImg() {
        System.out.println("Test query Milvus Service(txt2img) by embeddings:");
        MilvusService service = context.getBean(MilvusService.class);

        Map<String, String> args = new HashMap<>();
        args.put("collectionName", "txt_to_img_demo");
        args.put("outFields", "id");
        args.put("vectorField", "image_emb");
        service.setMilvusArgs(args);

        // Note the emb dim must be same with data stored in Milvus
        List<List<Float>> vectors = generateFloatVectors(1, 512);
        Map<Integer, List<SearchResultsWrapper.IDScore>> result = service.findByEmbeddingVectors(vectors, 5);
        System.out.println(result);
    }

    private List<List<Float>> generateFloatVectors(int count, int dimension) {
        Random ran = new Random(1234567890);
        List<List<Float>> vectors = Lists.newArrayList();
        for (int n = 0; n < count; ++n) {
            List<Float> vector = Lists.newArrayList();
            for (int i = 0; i < dimension; ++i) {
                vector.add(ran.nextFloat());
            }
            vectors.add(vector);
        }
        return vectors;
    }

    private List<List<Float>> generateFloatVectors() {
        List<List<Float>> vectors = Lists.newArrayList();

        // the 768 dim item sample
        String v1 = "0.45396071672439575, -0.22542467713356018, -0.7078282237052917, 0.005351922009140253, 0.099767304956913, -0.44279715418815613, -1.0261484384536743, 0.23563089966773987, -1.5961992740631104, -0.5587632656097412, -0.14583691954612732, -0.8990698456764221, -0.03202144056558609, -1.277194619178772, -0.6558062434196472, -0.21193362772464752, 0.02214740589261055, 0.3147466480731964, 0.24563223123550415, -0.7223032116889954, -0.23184293508529663, -0.6717422604560852, -0.2453237771987915, -0.2668493688106537, 0.18486636877059937, 0.6597995162010193, -0.4787231981754303, -0.3411640524864197, -0.44049766659736633, -0.9958179593086243, 1.2174910306930542, -0.22645464539527893, -0.7164020538330078, 0.08988014608621597, -1.1407909393310547, 0.38870498538017273, -0.24885721504688263, 1.3992605209350586, -0.6510639190673828, -0.7131631374359131, 0.5792143940925598, -0.20433558523654938, -0.3007623255252838, -0.14233198761940002, 0.8006675839424133, 0.3534839153289795, 0.40700870752334595, 0.16727063059806824, -0.15361744165420532, 0.4388487637042999, 0.29221105575561523, 5.616407871246338, -0.40708136558532715, 0.5584620237350464, -0.8778289556503296, 0.45917201042175293, -1.1341241598129272, -0.37445056438446045, -0.05796812102198601, -1.0355814695358276, 0.8216122388839722, -0.09432800114154816, 1.536928415298462, -0.5473815202713013, 0.4867522418498993, 0.18351013958454132, 0.20714087784290314, 0.5883209109306335, 0.25533145666122437, -0.6729830503463745, -0.006549303885549307, -0.25901076197624207, 0.4663187563419342, 0.6961797475814819, 0.5269835591316223, -0.37257033586502075, 0.3660552501678467, 0.08670218288898468, 0.04222479462623596, -0.15932975709438324, -0.3056860566139221, 1.3088186979293823, -0.7418187260627747, 0.06680626422166824, -0.25865641236305237, -0.14842307567596436, -0.7633294463157654, -0.9545206427574158, 0.2947665750980377, 1.0268930196762085, -1.3858938217163086, -0.3501761853694916, -0.5340178608894348, 0.06471413373947144, 0.3410576283931732, -0.3357231616973877, 0.7128798365592957, -0.28460603952407837, -0.00604236638173461, -0.41652750968933105, 0.45098361372947693, -0.12882816791534424, 0.019312789663672447, -0.5951293706893921, -0.10218826681375504, 0.19831590354442596, 0.2586207985877991, 0.6100096702575684, -0.21768036484718323, 0.012973031960427761, -0.2216944843530655, -0.04271715134382248, -0.3453143239021301, -0.35367944836616516, -1.2681829929351807, -0.7261069416999817, -0.17398479580879211, -0.7966687679290771, 0.4374055862426758, -0.016676582396030426, -0.48723989725112915, -0.11025295406579971, -0.12580284476280212, 0.2680254280567169, -0.8467956185340881, -0.13659892976284027, -0.0978088453412056, 0.23482269048690796, -0.6393059492111206, 0.6801612973213196, 0.7938153743743896, 0.4081241190433502, -0.3518829941749573, -0.2786822021007538, 0.09165728837251663, -0.2861272394657135, -0.7447338700294495, -0.03433603048324585, -1.0245827436447144, 0.026385728269815445, 0.6364550590515137, -0.44588547945022583, -0.8279715776443481, 0.3205845355987549, -0.09575439244508743, -1.2930831909179688, 0.3024573028087616, 0.9305884838104248, -0.009143523871898651, -0.5151926279067993, 0.20297826826572418, -0.5437697172164917, -0.04798787087202072, 0.3740179240703583, 0.4759384095668793, 0.43152230978012085, 0.12239988893270493, -0.033856481313705444, -0.6895148158073425, -0.399148166179657, 0.01791001483798027, -0.2011605054140091, -0.5081787109375, -0.5884726047515869, -0.6236153841018677, -0.20986782014369965, 0.04594776779413223, -0.4763299226760864, 0.6516847014427185, 1.2979613542556763, -0.5722696781158447, 0.7503501176834106, 0.5497811436653137, -0.20714369416236877, 0.6001092791557312, 0.3733534514904022, 0.3759412467479706, -0.7766485810279846, -0.7043375372886658, -2.024897336959839, -1.2995071411132812, 0.06370662152767181, -0.2990366816520691, -0.09190601110458374, 0.528980016708374, -0.4007043242454529, -0.8942004442214966, 0.32220691442489624, 0.2945502996444702, 0.9477272033691406, -0.479954332113266, -0.551918089389801, -0.7736573219299316, -0.46604007482528687, 0.2161029577255249, 0.732734203338623, -0.1227802112698555, 0.25280073285102844, -0.10123548656702042, 0.05442815646529198, -1.3820065259933472, -0.013179452158510685, 0.6094679236412048, 0.7304729223251343, 0.8954242467880249, 1.2425585985183716, -0.9562987089157104, -0.7179126143455505, 0.45344409346580505, 1.2209558486938477, 0.7372780442237854, -0.755447268486023, 0.017121948301792145, -0.4899780750274658, 0.8210601806640625, 0.013429015874862671, -0.372108519077301, 0.2866063416004181, -0.6149254441261292, -0.49106165766716003, 0.27203479409217834, 0.00605811458081007, 0.37862712144851685, 0.7419453263282776, 0.19439882040023804, 0.8169022798538208, -0.7800169587135315, -0.4973948895931244, 0.27419185638427734, -0.9105935096740723, 0.7717941999435425, 0.5834916830062866, 0.1664571315050125, 0.059429995715618134, 0.20379741489887238, 0.25014179944992065, -0.03324022889137268, 1.2336875200271606, -0.44672703742980957, 0.7844028472900391, -0.38105177879333496, -0.11484376341104507, -0.0829290971159935, 0.0993850901722908, -0.30770668387413025, 0.5040580630302429, -0.4124574661254883, -0.7262893319129944, -0.5051444172859192, -0.48193106055259705, 0.3684680759906769, 0.37940242886543274, 0.33584773540496826, 1.4120947122573853, 0.0747503787279129, -0.48854899406433105, -0.7320857048034668, 0.7441822290420532, -0.0882757157087326, -0.21618513762950897, 0.03524135425686836, -0.4563182592391968, 1.0995213985443115, 0.16075429320335388, 0.22037847340106964, -0.055005185306072235, -0.4731628894805908, 0.04688423499464989, -0.2666163742542267, -0.020805001258850098, 0.4574261009693146, -0.22008082270622253, -0.12884581089019775, -0.6048802733421326, -0.3023937940597534, -0.4644797146320343, 0.6834802627563477, 0.8540149927139282, 0.7536017894744873, 0.6697497963905334, -0.41873472929000854, -0.43966665863990784, -0.45284977555274963, 0.6314351558685303, -0.7252306938171387, -0.09262080490589142, 0.04776127263903618, 0.7171989679336548, -0.4139852523803711, -0.5627545118331909, -0.43755367398262024, 0.41501009464263916, -0.18118855357170105, -0.5969259142875671, 2.084872007369995, -0.2590540051460266, -0.9717243313789368, -0.34118273854255676, 0.32504889369010925, -0.27489185333251953, -1.2269268035888672, -0.5764029622077942, -1.0724256038665771, 0.8024383783340454, 0.4884457290172577, -0.14187496900558472, 0.05823074281215668, 1.261179804801941, 0.5525150299072266, -0.45316174626350403, -1.0030722618103027, -1.3473976850509644, 0.5125463604927063, 0.37202244997024536, -1.3611671924591064, -0.35262331366539, 1.1450042724609375, 0.1292506754398346, 1.3844949007034302, -0.1121668890118599, 0.36855483055114746, 0.963092565536499, -0.2791167199611664, 1.422154426574707, -0.7321933507919312, 0.6210702657699585, -0.07950551062822342, -0.059917204082012177, 0.24276433885097504, 0.33481597900390625, -0.08795100450515747, -0.382163405418396, 1.1942369937896729, 0.4653305411338806, 0.18674981594085693, -0.6353798508644104, -0.31579139828681946, 0.5985535979270935, -0.7487959861755371, -1.043463945388794, -0.49987322092056274, 0.5850246548652649, 0.5391435623168945, 0.5300890207290649, -0.9748578667640686, -0.7012567520141602, -0.19833526015281677, 1.5940672159194946, -0.4903988540172577, -0.4905678331851959, 1.0904065370559692, -0.20565353333950043, -0.6770005822181702, -0.11575789004564285, -0.4593571126461029, 0.09530121833086014, 0.31970950961112976, 0.06843734532594681, 0.6982269287109375, 0.40778589248657227, 0.387994647026062, 0.6212781071662903, 0.6356711387634277, 0.4715994596481323, -0.7165288329124451, 0.4832432270050049, -0.5043458938598633, 1.082849383354187, 0.521824061870575, -0.01811046153306961, -0.2681639492511749, 0.2986237406730652, 0.13939552009105682, -0.26662635803222656, 1.3125801086425781, -1.0746759176254272, 0.06319630891084671, -0.7790918946266174, -0.6525376439094543, 0.32536575198173523, -0.20735125243663788, -0.514907717704773, 0.26458263397216797, -1.1204068660736084, 0.6111680865287781, 0.23932339251041412, 0.14863669872283936, -0.9770679473876953, 0.24383017420768738, 0.05449282005429268, -0.46218299865722656, 0.9058054089546204, -0.8333122134208679, -0.7690218687057495, 0.5780305862426758, -0.23399797081947327, -0.23939988017082214, -0.23220306634902954, -0.31211626529693604, 0.422507643699646, 0.6318041682243347, -0.08876007795333862, -0.12159139662981033, -0.03503286466002464, 1.489380955696106, -0.3356373906135559, 0.6841373443603516, -0.2651520371437073, 0.36282819509506226, -0.5326091051101685, 0.20317380130290985, 0.04399258643388748, -0.42937585711479187, -0.6042398810386658, 0.7726759910583496, 0.15367873013019562, 0.11151215434074402, -0.43888968229293823, 0.4912474453449249, -0.8660299181938171, 0.0497574657201767, -0.37002381682395935, -0.12494568526744843, -0.43693873286247253, 0.9080032706260681, -0.411288321018219, -0.3323262333869934, 0.1037697046995163, -0.7895727753639221, -0.845513641834259, 0.03335167095065117, 0.5194165110588074, -0.6507505774497986, -0.46301358938217163, 1.2501775026321411, 1.0236891508102417, -0.5799307227134705, 0.28646016120910645, -0.857353687286377, -0.6262983083724976, -0.6035915017127991, 0.33139675855636597, -0.16342191398143768, 0.4576944410800934, 0.021168839186429977, 0.3754502534866333, 0.08704888075590134, -0.66413813829422, 0.030307158827781677, 0.8539645671844482, 0.9228533506393433, -0.163674995303154, 0.5276708602905273, 0.5367852449417114, -0.8561933636665344, 0.05938183516263962, 0.4166203439235687, 0.07718710601329803, 0.0791870579123497, 0.3313346207141876, -0.20645637810230255, -0.38994842767715454, -1.4917761087417603, -0.131621316075325, 0.5897359251976013, -0.11883282661437988, 0.7183640599250793, 0.18194428086280823, 1.1455413103103638, 0.30458760261535645, 1.0985621213912964, -1.4297308921813965, 0.4153004586696625, -0.9877591729164124, -0.6017285585403442, 0.3995997905731201, -0.255461186170578, 0.1242261752486229, -0.17907218635082245, -0.28974175453186035, -0.36513862013816833, 0.5625249147415161, 0.061375413089990616, 0.3000588119029999, -0.5633060336112976, 0.3878535032272339, 1.2093359231948853, 0.4638512432575226, 0.7879377007484436, -0.08703671395778656, -0.44771820306777954, 0.37783288955688477, 0.9713508486747742, -0.9182921648025513, 0.42940929532051086, -0.7756760120391846, -0.1893715262413025, -0.47076886892318726, 0.020379705354571342, -0.16998805105686188, -0.04254290089011192, -1.3691987991333008, 0.66277015209198, 0.16688492894172668, 0.07466834783554077, -0.3476543426513672, 0.2220253050327301, -0.18947410583496094, -0.7951254844665527, 1.0230989456176758, 1.332826018333435, -0.3289775848388672, -1.0122594833374023, -0.8126025199890137, 1.3762487173080444, -0.16422820091247559, -0.731784999370575, 0.4165458679199219, -0.7173048257827759, -0.018902264535427094, 1.131943941116333, -0.1201290488243103, 0.2744821310043335, 0.34442591667175293, 0.09769704192876816, 0.21651297807693481, -0.43463319540023804, -0.16688652336597443, -0.8658238053321838, 0.4528611898422241, 0.11475377529859543, 0.4561823308467865, -0.12489493936300278, 0.48395583033561707, -0.24696795642375946, 0.7038353085517883, 0.8679671287536621, -0.4792214035987854, 0.3236652612686157, -0.47966933250427246, 0.029581598937511444, 0.06212874874472618, 0.5051004886627197, -1.0812510251998901, 0.031061453744769096, -0.7202182412147522, -0.018806785345077515, -0.3173049986362457, 0.4921391010284424, 0.6684141159057617, -1.2626428604125977, 0.8840476870536804, -0.3095659911632538, 0.33974185585975647, -0.49973875284194946, 0.26575541496276855, -0.6190115809440613, -0.5137141942977905, 0.33725205063819885, -0.06859191507101059, 0.045574851334095, 0.5048948526382446, -0.23238804936408997, -0.38060876727104187, -0.1930721551179886, -0.23841021955013275, 0.08908301591873169, 0.12322120368480682, -0.38467398285865784, 0.7694090008735657, 0.5110427141189575, 0.9530891180038452, -0.33499693870544434, 1.2310171127319336, 0.482582151889801, 0.2141256481409073, -0.14113610982894897, 0.2278682142496109, -0.06282488256692886, 0.7478843331336975, 0.10061938315629959, 0.9957262873649597, 0.5552526116371155, -0.5718789100646973, -0.4930558502674103, -0.7617856860160828, 0.05525663495063782, -0.10765960812568665, -0.36520737409591675, 0.4148786962032318, -0.2919744849205017, 0.21116285026073456, -0.49552637338638306, 0.35413095355033875, 0.3037145137786865, -0.2928258180618286, 0.7512683272361755, -0.1519956886768341, -0.9098075032234192, 0.5465769171714783, 0.49209415912628174, -1.4183824062347412, -0.6288448572158813, 0.2487228661775589, 1.2889615297317505, -2.294271230697632, 1.266502022743225, -0.4096996784210205, 0.05961006507277489, -1.6010627746582031, -0.42174237966537476, 0.8333172798156738, 0.23063351213932037, -0.8629800081253052, 0.8353806734085083, 0.8797457814216614, 0.306712806224823, 0.9531974792480469, -1.028926134109497, -0.23852598667144775, 0.17969198524951935, -0.5897449851036072, -0.4731694459915161, 0.4651114344596863, -0.41618213057518005, -0.10092645138502121, 2.224579334259033, 1.5131659507751465, 1.1030185222625732, -0.25879380106925964, 0.3098903000354767, -0.5637789368629456, -0.27684274315834045, 0.18936599791049957, -0.2992568612098694, 0.8704861402511597, -0.6445021629333496, -0.8920134902000427, -0.6969581246376038, -0.7500591278076172, 0.6357503533363342, -0.557075023651123, -0.9567848443984985, -0.9096843600273132, -0.16877003014087677, -0.662460446357727, 0.6795646548271179, 0.008216222748160362, -0.20440351963043213, -0.25272977352142334, 0.7854836583137512, -0.5106655359268188, -0.7067935466766357, -0.3510080873966217, 1.5365452766418457, -0.3481195867061615, 0.8542337417602539, -0.0027485419996082783, -0.35986900329589844, -0.2539672553539276, 0.2276572585105896, -0.05045995116233826, -0.056405793875455856, 0.3001209795475006, 0.24282491207122803, 0.22456800937652588, 0.44367820024490356, -0.5599551200866699, 0.06695316731929779, 0.8462013602256775, 1.2013272047042847, -0.3331170380115509, 0.802865743637085, -1.081921100616455, -0.5807102918624878, 0.9285374283790588, 0.5885787010192871, -0.036827415227890015, -0.6775050163269043, 0.2877728343009949, -0.5121967196464539, -0.13023078441619873, 1.6362543106079102, -0.9530014991760254, 0.4283394515514374, 0.09461215138435364, -0.23909339308738708, 0.020813049748539925, -0.4120487868785858, 0.24256080389022827, 1.195908784866333, 1.4630321264266968, -0.021146314218640327, -1.0211725234985352, 0.26190006732940674, 1.9797816276550293, 0.6008377075195312, -0.4801701307296753, -1.051792860031128, 0.05255676060914993, 0.42346855998039246, 0.7933213710784912, -0.1533268541097641, -0.22273890674114227, 0.459381103515625, 0.08287398517131805, -0.7344949841499329, -0.2541603744029999, -0.20246097445487976, 0.5138893723487854, -0.62841796875, 0.9887738227844238, -0.43961453437805176, 0.42386752367019653, 0.37006238102912903, -0.8966240286827087, -0.2536673843860626, -1.2379429340362549, 0.4899511933326721, 1.0182100534439087, -0.867165744304657, -1.1685378551483154, -0.25216418504714966, -0.24157622456550598, -0.5368327498435974, -0.12492154538631439, -1.233229637145996, 0.05595606938004494, 0.006861455272883177, 0.3318190276622772, 0.7410929203033447, 0.1228346973657608, -0.6574109196662903, -0.49938279390335083, -0.9742574095726013, 0.4223661422729492, 0.7915825247764587, -0.03301789239048958, 0.7133688926696777, 0.35948362946510315, -0.8342593312263489, 0.33601975440979004, 1.3182188272476196, 0.2918766736984253, 0.3364546298980713, 0.002529048826545477, 0.12814997136592865, -0.48749199509620667, -0.9852904081344604, 0.19198602437973022, 0.6475247144699097, 0.07945368438959122, 0.10539151728153229, 0.44700464606285095, -0.38266462087631226, -0.6895318627357483, -0.32702237367630005, -1.1074837446212769, -0.18005158007144928, 0.6774888038635254, 0.18714947998523712, -0.00036118310526944697, 0.3734060525894165, 0.2625422179698944, -0.5396274924278259, -0.16114087402820587, -0.5039079189300537, -0.03246808797121048";
        List<Float> vector = Lists.newArrayList();
        String[] buff = v1.split(",");
        for (int i=0; i< buff.length; i++) {
            vector.add(Float.parseFloat(buff[i].trim()));
        }
        vectors.add(vector);

        return vectors;
    }
}