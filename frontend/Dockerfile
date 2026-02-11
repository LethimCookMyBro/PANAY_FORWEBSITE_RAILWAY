# ใช้ Node.js เวอร์ชัน 18-alpine เป็น Base Image
FROM node:18-alpine

# กำหนด Working Directory ภายใน Container
WORKDIR /app

# คัดลอกแค่ package.json และ package-lock.json (ถ้ามี) ก่อน
# เพื่อใช้ประโยชน์จาก Docker Cache
COPY package.json ./
# COPY package-lock.json ./

# ติดตั้ง Dependencies ทั้งหมดภายใน Container
# ขั้นตอนนี้จะสร้าง node_modules ที่ถูกต้องสำหรับ Linux
RUN npm install

# คัดลอกไฟล์โค้ดที่เหลือทั้งหมดเข้ามาใน Container
# Docker จะไม่คัดลอก node_modules จากเครื่องคุณ เพราะเราจะกำหนดใน .dockerignore
COPY . .

# เปิดพอร์ต 5173 ให้สามารถเข้าถึงได้
EXPOSE 5173

# คำสั่งที่จะรัน Vite Dev Server เมื่อ Container เริ่มทำงาน
# '--host' ทำให้สามารถเข้าถึงได้จากภายนอก Container
CMD ["npm", "run", "dev", "--", "--host"]