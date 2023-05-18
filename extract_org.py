from bs4 import BeautifulSoup
html_doc = """<select class="form-select" id="select-org" onchange="selectChangeOrg(this)" data-dashlane-rid="316c55616c758781"
    data-form-type="other">
    <option selected="" value="ทั้งหมด">ทั้งหมด</option>
    <option value="18055">เขตคลองเตย</option>
    <option value="18056">เขตคลองสาน</option>
    <option value="18057">เขตคลองสามวา</option>
    <option value="18058">เขตคันนายาว</option>
    <option value="18059">เขตจตุจักร</option>
    <option value="18013">เขตจอมทอง</option>
    <option value="18014">เขตดอนเมือง</option>
    <option value="18015">เขตดินแดง</option>
    <option value="17370">เขตดุสิต</option>
    <option value="18016">เขตตลิ่งชัน</option>
    <option value="18017">เขตทวีวัฒนา</option>
    <option value="18018">เขตทุ่งครุ</option>
    <option value="18019">เขตธนบุรี</option>
    <option value="18021">เขตบางกอกน้อย</option>
    <option value="18022">เขตบางกอกใหญ่</option>
    <option value="18020">เขตบางกะปิ</option>
    <option value="18023">เขตบางขุนเทียน</option>
    <option value="18024">เขตบางเขน</option>
    <option value="18025">เขตบางคอแหลม</option>
    <option value="18026">เขตบางแค</option>
    <option value="8371">เขตบางซื่อ</option>
    <option value="18027">เขตบางนา</option>
    <option value="18028">เขตบางบอน</option>
    <option value="18029">เขตบางพลัด</option>
    <option value="18030">เขตบางรัก</option>
    <option value="18031">เขตบึงกุ่ม</option>
    <option value="18032">เขตปทุมวัน</option>
    <option value="18033">เขตประเวศ</option>
    <option value="18034">เขตป้อมปราบศัตรูพ่าย</option>
    <option value="18035">เขตพญาไท</option>
    <option value="18037">เขตพระโขนง</option>
    <option value="18036">เขตพระนคร</option>
    <option value="18038">เขตภาษีเจริญ</option>
    <option value="18039">เขตมีนบุรี</option>
    <option value="18040">เขตยานนาวา</option>
    <option value="18041">เขตราชเทวี</option>
    <option value="18042">เขตราษฎร์บูรณะ</option>
    <option value="18043">เขตลาดกระบัง</option>
    <option value="17952">เขตลาดพร้าว</option>
    <option value="18044">เขตวังทองหลาง</option>
    <option value="18045">เขตวัฒนา</option>
    <option value="18050">เขตสวนหลวง</option>
    <option value="18046">เขตสะพานสูง</option>
    <option value="18049">เขตสัมพันธวงศ์</option>
    <option value="18047">เขตสาทร</option>
    <option value="18048">เขตสายไหม</option>
    <option value="18052">เขตหนองแขม</option>
    <option value="18051">เขตหนองจอก</option>
    <option value="18053">เขตหลักสี่</option>
    <option value="18054">เขตห้วยขวาง</option>
    <option value="18117">สำนักการคลัง กทม.</option>
    <option value="18109">สำนักการจราจรและขนส่ง กรุงเทพมหานคร (สจส.)</option>
    <option value="18118">สำนักการแพทย์ กทม.</option>
    <option value="18119">สำนักการโยธา กทม.</option>
    <option value="18067">สำนักการระบายน้ำ กทม.</option>
    <option value="18126">สำนักการวางผังและพัฒนาเมือง กทม.</option>
    <option value="18121">สำนักการศึกษา กทม.</option>
    <option value="18122">สำนักงบประมาณกรุงเทพมหานคร</option>
    <option value="18116">สำนักงานคณะกรรมการข้าราชการกรุงเทพมหานคร</option>
    <option value="18123">สำนักเทศกิจ กทม.</option>
    <option value="18010">สำนักป้องกันและบรรเทาสาธารณภัย กทม.</option>
    <option value="18125">สำนักพัฒนาสังคม กทม.</option>
    <option value="18127">สำนักยุทธศาสตร์และประเมินผล กทม.</option>
    <option value="18128">สำนักวัฒนธรรม กีฬาและการท่องเที่ยว กทม.</option>
    <option value="18129">สำนักสิ่งแวดล้อม กทม.</option>
    <option value="18130">สำนักอนามัย กทม.</option>
</select>"""

soup = BeautifulSoup(html_doc, 'html.parser')

# print(soup.prettify())
op_list = soup.find_all('option')
with open('organizations.txt', 'w', encoding='utf-8') as f:
    for op in op_list:
        f.write(op.string + '\n')