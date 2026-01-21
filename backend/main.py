# backend/main.py
from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator
from typing import List, Optional, Dict
import uvicorn
from datetime import datetime
import os
from pathlib import Path
import jwt
import hashlib
import uuid
import hmac
import json
from uuid import uuid4
import secrets
import shutil
import aiofiles

# Добавляем PostgreSQL
import asyncpg
from asyncpg.pool import Pool
import asyncio
from contextlib import asynccontextmanager

# Загрузка переменных окружения
from dotenv import load_dotenv

load_dotenv()

from fastapi.templating import Jinja2Templates
from fastapi import Request
from pathlib import Path
import os

# ========== НАСТРОЙКА ==========
BASE_DIR = Path(__file__).parent.parent

# Настройка Jinja2 с проверкой пути
templates_path = BASE_DIR / "templates"

# Проверяем путь в разных вариантах (для локальной разработки и Docker)
if not templates_path.exists():
    # Попробуем другой путь (для Docker)
    templates_path = Path("/app/templates")  # Docker путь
    if not templates_path.exists():
        # Создаем папку и базовые файлы
        templates_path = BASE_DIR / "templates"
        templates_path.mkdir(exist_ok=True)
        print(f"📁 Создана папка шаблонов: {templates_path}")

print(f"🔍 Путь к шаблонам: {templates_path.absolute()}")
print(f"📁 Содержимое: {list(templates_path.glob('*.html'))}")

# Настройка Jinja2
templates = Jinja2Templates(directory=str(templates_path))

# ========== НАСТРОЙКА ==========
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = STATIC_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Создание директорий
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "images").mkdir(exist_ok=True)
(STATIC_DIR / "favicon").mkdir(exist_ok=True)

# Создаем изображение по умолчанию если его нет
DEFAULT_IMAGE_PATH = STATIC_DIR / "images" / "product_default.jpg"
if not DEFAULT_IMAGE_PATH.exists():
    try:
        # Создаем простую текстовую заглушку
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new('RGB', (400, 300), color='#f0f0f0')
        draw = ImageDraw.Draw(img)

        # Пробуем использовать стандартный шрифт
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        draw.text((100, 120), "NO IMAGE", fill='#888888', font=font)
        draw.text((80, 160), "Изображение товара", fill='#666666', font=font)
        img.save(DEFAULT_IMAGE_PATH)
        print("✅ Создано изображение-заглушка")
    except Exception as e:
        print(f"⚠️ Не удалось создать изображение-заглушку: {e}")

# Пароль для админки (из переменных окружения или по умолчанию)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_USERNAME = "admin"  # Фиксированное имя пользователя для админки

# PostgreSQL connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/scooter_shop")


# ========== МОДЕЛИ ==========
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str
    phone: Optional[str] = None

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        if len(v) < 3:
            raise ValueError('Имя пользователя должно содержать минимум 3 символа')
        if len(v) > 50:
            raise ValueError('Имя пользователя должно содержать не более 50 символов')
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError('Пароль должен содержать минимум 6 символов')
        return v


class UserLogin(BaseModel):
    username: str
    password: str


class Product(BaseModel):
    id: int
    name: str
    category: str
    price: float
    description: str
    image_url: str
    stock: int = 0
    featured: bool = False


class CartItem(BaseModel):
    product_id: int
    quantity: int


class CartUpdate(BaseModel):
    product_id: int
    quantity: int


# Модели для админки
class AdminLogin(BaseModel):
    username: str
    password: str


class ProductCreate(BaseModel):
    name: str
    category: str
    price: float
    description: str
    stock: int = 0
    featured: bool = False


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    description: Optional[str] = None
    stock: Optional[int] = None
    featured: Optional[bool] = None


# ========== АУТЕНТИФИКАЦИЯ ==========
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
security = HTTPBearer()


# Альтернативный хэшер на основе PBKDF2 (без bcrypt)
class PasswordHasher:
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Генерация соли и хэша пароля"""
        # Генерируем случайную соль
        salt = secrets.token_hex(16)
        # Используем PBKDF2 с SHA256 для хэширования
        iterations = 100000
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations
        )
        # Формат: алгоритм:итерации:соль:хэш
        return f"pbkdf2_sha256:{iterations}:{salt}:{key.hex()}"

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Проверка пароля"""
        try:
            # Разбираем сохраненный хэш
            parts = hashed_password.split(':')
            if len(parts) != 4:
                return False

            algorithm, iterations_str, salt, stored_hash = parts
            if algorithm != 'pbkdf2_sha256':
                return False

            iterations = int(iterations_str)

            # Вычисляем хэш для введенного пароля
            key = hashlib.pbkdf2_hmac(
                'sha256',
                plain_password.encode('utf-8'),
                salt.encode('utf-8'),
                iterations
            )

            # Сравниваем хэши безопасным способом
            return hmac.compare_digest(key.hex(), stored_hash)

        except (ValueError, AttributeError, Exception):
            return False


# Создаем экземпляр хэшера
hasher = PasswordHasher()


def create_access_token(data: dict):
    """Создание JWT токена"""
    to_encode = data.copy()
    # Убеждаемся, что все значения сериализуемы
    for key, value in to_encode.items():
        if isinstance(value, (uuid.UUID, datetime)):
            to_encode[key] = str(value)

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("user_id")
        if user_id is None:
            return None
        return user_id
    except:
        return None


def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Проверка админских прав"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        is_admin = payload.get("is_admin")
        if not is_admin:
            raise HTTPException(status_code=403, detail="Доступ запрещен")
        return payload
    except:
        raise HTTPException(status_code=401, detail="Не авторизован")


# ========== POSTGRESQL БАЗА ДАННЫХ ==========
class Database:
    def __init__(self):
        self.pool: Optional[Pool] = None

    async def connect(self):
        """Подключение к базе данных"""
        try:
            self.pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            await self.init_database()
            print("✅ База данных подключена успешно")
        except Exception as e:
            print(f"❌ Ошибка подключения к базе данных: {e}")
            raise

    async def disconnect(self):
        """Отключение от базы данных"""
        if self.pool:
            await self.pool.close()
            print("✅ Подключение к базе данных закрыто")

    async def init_database(self):
        """Инициализация таблиц"""
        async with self.pool.acquire() as connection:
            try:
                # Создание таблицы пользователей
                await connection.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id UUID PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(100) UNIQUE NOT NULL,
                        full_name VARCHAR(100) NOT NULL,
                        phone VARCHAR(20),
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_admin BOOLEAN DEFAULT FALSE
                    )
                ''')

                # Создание таблицы продуктов
                await connection.execute('''
                    CREATE TABLE IF NOT EXISTS products (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(200) NOT NULL,
                        category VARCHAR(50) NOT NULL,
                        price DECIMAL(10, 2) NOT NULL,
                        description TEXT NOT NULL,
                        image_url VARCHAR(500) NOT NULL,
                        stock INTEGER DEFAULT 0,
                        featured BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Создание таблицы корзин
                await connection.execute('''
                    CREATE TABLE IF NOT EXISTS cart_items (
                        id SERIAL PRIMARY KEY,
                        user_id UUID NOT NULL,
                        product_id INTEGER NOT NULL,
                        quantity INTEGER NOT NULL CHECK (quantity > 0),
                        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, product_id),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
                    )
                ''')

                # Создание демо-пользователя если не существует
                demo_user_exists = await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM users WHERE username = 'demo')"
                )

                if not demo_user_exists:
                    user_id = str(uuid4())
                    password_hash = hasher.get_password_hash("demo123")
                    await connection.execute('''
                        INSERT INTO users (id, username, email, full_name, phone, password_hash, is_admin)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ''', user_id, 'demo', 'demo@scooterparts.ru', 'Демо Пользователь',
                                             '+79991234567', password_hash, False)

                # Создание админ пользователя если не существует
                admin_exists = await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM users WHERE username = 'admin')"
                )

                if not admin_exists:
                    admin_id = str(uuid4())
                    password_hash = hasher.get_password_hash(ADMIN_PASSWORD)
                    await connection.execute('''
                        INSERT INTO users (id, username, email, full_name, password_hash, is_admin)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    ''', admin_id, 'admin', 'admin@scooterparts.ru', 'Администратор',
                                             password_hash, True)

                # Создание демо-товаров если их нет
                products_count = await connection.fetchval("SELECT COUNT(*) FROM products")

                if products_count == 0:
                    demo_products = [
                        ("Аккумулятор Premium 36V 15Ah", "batteries", 16500.00,
                         "Высокоёмкий литий-ионный аккумулятор с системой защиты BMS. Гарантия 24 месяца.",
                         "/static/images/battery.jpg", 8, True),
                        ("Мотор-колесо Ultra 500W", "motors", 12500.00,
                         "Бесщёточный мотор с прямым приводом. Максимальная скорость 45 км/ч.",
                         "/static/images/motor.jpg", 5, True),
                        ("Контроллер Smart 36V", "electronics", 4900.00,
                         "Интеллектуальный контроллер с Bluetooth и мобильным приложением.",
                         "/static/images/controller.jpg", 15, False),
                        ("Дисплей Color LCD", "electronics", 3200.00,
                         "Цветной LCD дисплей с подсветкой и индикацией всех параметров.",
                         "/static/images/display.jpg", 12, True),
                        ("Тормозные диски Premium", "brakes", 2200.00,
                         "Вентилируемые тормозные диски из нержавеющей стали.",
                         "/static/images/brakes.jpg", 25, False),
                        ("Колесо 10\" All-Terrain", "tires", 1800.00,
                         "Пневматическое колесо для бездорожья с усиленными стенками.",
                         "/static/images/wheel.jpg", 20, False),
                        ("Тормозные колодки Premium", "brakes", 1200.00,
                         "Керамические тормозные колодки для дисковых тормозов.",
                         "/static/images/brake-pads.jpg", 30, True),
                        ("Руль алюминиевый", "accessories", 2500.00,
                         "Алюминиевый руль с резиновыми накладками.",
                         "/static/images/handlebar.jpg", 15, False)
                    ]

                    for product in demo_products:
                        await connection.execute('''
                            INSERT INTO products (name, category, price, description, image_url, stock, featured)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ''', *product)

                print("✅ Таблицы базы данных инициализированы")

            except Exception as e:
                print(f"❌ Ошибка инициализации базы данных: {e}")
                raise


# Создаем экземпляр базы данных
db = Database()


# ========== Lifespan Events (замена устаревшего on_event) ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db.connect()
    print("✅ Приложение запущено")
    yield
    # Shutdown
    await db.disconnect()
    print("✅ Приложение остановлено")


# ========== СОЗДАНИЕ ПРИЛОЖЕНИЯ ==========
app = FastAPI(
    title="Scooter Parts Shop",
    description="Премиум запчасти для электросамокатов",
    version="4.5.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтирование статических файлов
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ========== API ЭНДПОИНТЫ ==========
@app.post("/api/register")
async def register(user_data: UserRegister):
    """Регистрация нового пользователя"""
    try:
        async with db.pool.acquire() as connection:
            # Проверяем, существует ли пользователь
            user_exists = await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM users WHERE username = $1 OR email = $2)",
                user_data.username, user_data.email
            )

            if user_exists:
                raise HTTPException(status_code=400, detail="Имя пользователя или email уже используется")

            # Создаем нового пользователя
            user_id = str(uuid4())  # Уже строка
            password_hash = hasher.get_password_hash(user_data.password)

            await connection.execute('''
                INSERT INTO users (id, username, email, full_name, phone, password_hash)
                VALUES ($1, $2, $3, $4, $5, $6)
            ''', user_id, user_data.username, user_data.email,
                                     user_data.full_name, user_data.phone, password_hash)

            # Создаем токен
            access_token = create_access_token({"user_id": user_id})

            return {
                "message": "Регистрация успешна",
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "id": user_id,
                    "username": user_data.username,
                    "email": user_data.email,
                    "full_name": user_data.full_name
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка регистрации: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.post("/api/login")
async def login(login_data: UserLogin):
    """Вход пользователя"""
    try:
        async with db.pool.acquire() as connection:
            print(f"🔍 Ищем пользователя: {login_data.username}")

            # Ищем пользователя
            user = await connection.fetchrow(
                "SELECT id, username, email, full_name, password_hash FROM users WHERE username = $1",
                login_data.username
            )

            if not user:
                print(f"❌ Пользователь не найден: {login_data.username}")
                raise HTTPException(status_code=401, detail="Неверное имя пользователя или пароль")

            print(f"✅ Пользователь найден: {user['username']}")
            print(f"📝 Хэш пароля в базе: {user['password_hash'][:50]}...")

            # Проверяем пароль
            if not hasher.verify_password(login_data.password, user['password_hash']):
                print(f"❌ Неверный пароль для пользователя: {login_data.username}")
                raise HTTPException(status_code=401, detail="Неверное имя пользователя или пароль")

            print("✅ Пароль проверен успешно")

            # Создаем токен - конвертируем UUID в строку
            user_id = str(user['id'])  # Конвертируем UUID в строку
            access_token = create_access_token({"user_id": user_id})

            return {
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "id": user_id,  # Используем строку
                    "username": user['username'],
                    "email": user['email'],
                    "full_name": user['full_name']
                }
            }

    except HTTPException as he:
        print(f"❌ HTTPException в login: {he.detail}")
        raise
    except Exception as e:
        print(f"❌ Ошибка входа: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.get("/api/profile")
async def get_profile(user_id: str = Depends(get_current_user)):
    """Получить профиль пользователя"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")

    try:
        async with db.pool.acquire() as connection:
            user = await connection.fetchrow(
                "SELECT id, username, email, full_name, phone, created_at FROM users WHERE id = $1",
                user_id
            )

            if not user:
                raise HTTPException(status_code=404, detail="Пользователь не найден")

            # Конвертируем UUID в строку и Decimal в float
            user_dict = dict(user)
            user_dict['id'] = str(user_dict['id'])  # Конвертируем UUID в строку
            if isinstance(user_dict.get('created_at'), datetime):
                user_dict['created_at'] = user_dict['created_at'].isoformat()

            return user_dict

    except Exception as e:
        print(f"Ошибка получения профиля: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


# Корзина
@app.get("/api/cart")
async def get_cart(user_id: str = Depends(get_current_user)):
    """Получить корзину пользователя"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")

    try:
        async with db.pool.acquire() as connection:
            # Получаем товары в корзине
            cart_items = await connection.fetch('''
                SELECT ci.product_id, ci.quantity, 
                       p.name, p.category, p.price, p.description, p.image_url, p.stock
                FROM cart_items ci
                JOIN products p ON ci.product_id = p.id
                WHERE ci.user_id = $1
                ORDER BY ci.added_at DESC
            ''', user_id)

            cart_with_details = []
            total = 0

            for item in cart_items:
                item_total = item['price'] * item['quantity']
                total += item_total
                cart_with_details.append({
                    "product_id": item['product_id'],
                    "quantity": item['quantity'],
                    "product": {
                        "id": item['product_id'],
                        "name": item['name'],
                        "category": item['category'],
                        "price": float(item['price']),
                        "description": item['description'],
                        "image_url": item['image_url'],
                        "stock": item['stock']
                    },
                    "item_total": item_total
                })

            return {
                "items": cart_with_details,
                "total": total,
                "items_count": len(cart_items)
            }

    except Exception as e:
        print(f"Ошибка получения корзины: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.post("/api/cart")
async def add_to_cart(cart_item: CartUpdate, user_id: str = Depends(get_current_user)):
    """Добавить товар в корзину"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")

    if cart_item.quantity <= 0:
        raise HTTPException(status_code=400, detail="Количество должно быть больше 0")

    try:
        async with db.pool.acquire() as connection:
            # Проверяем, есть ли товар
            product = await connection.fetchrow(
                "SELECT id, stock FROM products WHERE id = $1",
                cart_item.product_id
            )

            if not product:
                raise HTTPException(status_code=404, detail="Товар не найден")

            # Проверяем наличие на складе
            if product['stock'] < cart_item.quantity:
                raise HTTPException(status_code=400, detail="Недостаточно товара на складе")

            # Добавляем или обновляем товар в корзине
            await connection.execute('''
                INSERT INTO cart_items (user_id, product_id, quantity)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id, product_id) 
                DO UPDATE SET quantity = EXCLUDED.quantity
            ''', user_id, cart_item.product_id, cart_item.quantity)

            return {"message": "Товар добавлен в корзину"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка добавления в корзину: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.delete("/api/cart/{product_id}")
async def remove_from_cart(product_id: int, user_id: str = Depends(get_current_user)):
    """Удалить товар из корзины"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")

    try:
        async with db.pool.acquire() as connection:
            # Удаляем товар из корзины
            result = await connection.execute(
                "DELETE FROM cart_items WHERE user_id = $1 AND product_id = $2",
                user_id, product_id
            )

            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Товар не найден в корзине")

            return {"message": "Товар удален из корзины"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка удаления из корзины: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.delete("/api/cart")
async def clear_cart(user_id: str = Depends(get_current_user)):
    """Очистить корзину"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")

    try:
        async with db.pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM cart_items WHERE user_id = $1",
                user_id
            )

            return {"message": "Корзина очищена"}

    except Exception as e:
        print(f"Ошибка очистки корзины: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


# Продукты
@app.get("/api/products")
async def get_products(category: Optional[str] = None, featured: Optional[bool] = None):
    """Получить список товаров"""
    try:
        async with db.pool.acquire() as connection:
            query = "SELECT * FROM products WHERE 1=1"
            params = []

            if category:
                query += " AND category = $1"
                params.append(category)

            if featured is not None:
                if category:
                    query += " AND featured = $2"
                    params.append(featured)
                else:
                    query += " AND featured = $1"
                    params.append(featured)

            query += " ORDER BY id"

            products = await connection.fetch(query, *params)

            # Конвертируем Decimal в float для JSON
            result = []
            for product in products:
                product_dict = dict(product)
                product_dict['price'] = float(product['price'])
                result.append(product_dict)

            return result

    except Exception as e:
        print(f"Ошибка получения товаров: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/products/{product_id}")
async def get_product(product_id: int):
    """Получить товар по ID"""
    try:
        async with db.pool.acquire() as connection:
            product = await connection.fetchrow(
                "SELECT * FROM products WHERE id = $1",
                product_id
            )

            if not product:
                raise HTTPException(status_code=404, detail="Товар не найден")

            # Конвертируем Decimal в float
            product_dict = dict(product)
            product_dict['price'] = float(product['price'])

            return product_dict

    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка получения товара: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/categories")
async def get_categories():
    """Получить список категорий"""
    try:
        async with db.pool.acquire() as connection:
            categories = await connection.fetch('''
                SELECT category, COUNT(*) as count 
                FROM products 
                GROUP BY category 
                ORDER BY category
            ''')

            category_names = {
                "batteries": "Аккумуляторы",
                "motors": "Моторы",
                "electronics": "Электроника",
                "brakes": "Тормоза",
                "tires": "Колёса",
                "accessories": "Аксессуары"
            }

            result = []
            for cat in categories:
                result.append({
                    "id": cat['category'],
                    "name": category_names.get(cat['category'], cat['category']),
                    "count": cat['count']
                })

            return {"categories": result}

    except Exception as e:
        print(f"Ошибка получения категорий: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/stats")
async def get_stats():
    """Получить статистику магазина"""
    try:
        async with db.pool.acquire() as connection:
            total_products = await connection.fetchval("SELECT COUNT(*) FROM products")
            total_stock = await connection.fetchval("SELECT SUM(stock) FROM products")
            featured_products = await connection.fetchval(
                "SELECT COUNT(*) FROM products WHERE featured = true"
            )
            categories_count = await connection.fetchval(
                "SELECT COUNT(DISTINCT category) FROM products"
            )

            return {
                "total_products": total_products or 0,
                "total_orders": 0,
                "categories": categories_count or 0,
                "total_stock": total_stock or 0,
                "featured_products": featured_products or 0
            }

    except Exception as e:
        print(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


# Админские эндпоинты
@app.post("/api/admin/login")
async def admin_login(login_data: AdminLogin):
    """Вход в админку"""
    if login_data.username != ADMIN_USERNAME:
        raise HTTPException(status_code=401, detail="Неверные данные для входа")

    if login_data.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Неверные данные для входа")

    try:
        async with db.pool.acquire() as connection:
            # Получаем ID админа из базы
            admin_user = await connection.fetchrow(
                "SELECT id FROM users WHERE username = $1",
                ADMIN_USERNAME
            )

            if not admin_user:
                raise HTTPException(status_code=401, detail="Администратор не найден")

            # Создаем токен для админа - конвертируем UUID в строку
            admin_id = str(admin_user['id'])
            admin_token = create_access_token({
                "user_id": admin_id,
                "username": ADMIN_USERNAME,
                "is_admin": True
            })

            return {
                "access_token": admin_token,
                "token_type": "bearer",
                "user": {
                    "username": ADMIN_USERNAME,
                    "is_admin": True
                }
            }

    except Exception as e:
        print(f"Ошибка входа в админку: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/admin/stats")
async def get_admin_stats(admin=Depends(verify_admin)):
    """Получить расширенную статистику для админки"""
    try:
        async with db.pool.acquire() as connection:
            # Статистика пользователей
            total_users = await connection.fetchval("SELECT COUNT(*) FROM users")
            users_with_carts = await connection.fetchval(
                "SELECT COUNT(DISTINCT user_id) FROM cart_items"
            )

            # Статистика товаров
            total_products = await connection.fetchval("SELECT COUNT(*) FROM products")
            in_stock = await connection.fetchval(
                "SELECT COUNT(*) FROM products WHERE stock > 0"
            )
            out_of_stock = await connection.fetchval(
                "SELECT COUNT(*) FROM products WHERE stock = 0"
            )
            featured = await connection.fetchval(
                "SELECT COUNT(*) FROM products WHERE featured = true"
            )

            # Статистика корзин
            total_carts = total_users  # У каждого пользователя есть корзина
            empty_carts = total_users - users_with_carts

            return {
                "users": {
                    "total": total_users,
                    "with_carts": users_with_carts,
                    "without_carts": empty_carts
                },
                "products": {
                    "total": total_products,
                    "in_stock": in_stock,
                    "out_of_stock": out_of_stock,
                    "featured": featured
                },
                "carts": {
                    "total": total_carts,
                    "empty": empty_carts,
                    "with_items": users_with_carts
                }
            }

    except Exception as e:
        print(f"Ошибка получения админской статистики: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.post("/api/admin/products")
async def create_product(
        request: Request,
        admin=Depends(verify_admin)
):
    """Создать новый товар с возможностью загрузки изображения - УНИВЕРСАЛЬНАЯ ВЕРСИЯ"""
    try:
        form_data = await request.form()

        # Отладочная информация
        print("📥 Получены данные формы:")
        for key, value in form_data.items():
            if key != 'image_file':
                print(f"  {key}: {value}")

        # Извлекаем данные из формы
        name = form_data.get("name")
        category = form_data.get("category")
        price_str = form_data.get("price")
        description = form_data.get("description")
        stock_str = form_data.get("stock")
        featured_str = form_data.get("featured")
        image_url = form_data.get("image_url")
        image_file = form_data.get("image_file")

        # Преобразуем типы
        if not name:
            raise HTTPException(status_code=400, detail="Название товара обязательно")
        name = str(name).strip()

        if not category:
            raise HTTPException(status_code=400, detail="Категория товара обязательна")
        category = str(category).strip()

        if not price_str:
            raise HTTPException(status_code=400, detail="Цена товара обязательна")
        try:
            price = float(price_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Неверный формат цены. Используйте числа")

        if not description:
            raise HTTPException(status_code=400, detail="Описание товара обязательно")
        description = str(description).strip()

        if not stock_str:
            stock = 0
        else:
            try:
                stock = int(stock_str)
            except ValueError:
                raise HTTPException(status_code=400, detail="Неверный формат количества")

        if not featured_str:
            featured = False
        else:
            featured = featured_str.lower() == 'true'

        if image_url:
            image_url = str(image_url).strip()

        # Валидация данных
        if len(name) < 3:
            raise HTTPException(
                status_code=400,
                detail="Название товара должно содержать минимум 3 символа"
            )

        if price <= 0:
            raise HTTPException(
                status_code=400,
                detail="Цена должна быть больше 0"
            )

        if len(description) < 10:
            raise HTTPException(
                status_code=400,
                detail="Описание должно содержать минимум 10 символов"
            )

        if stock < 0:
            raise HTTPException(
                status_code=400,
                detail="Количество не может быть отрицательным"
            )

        async with db.pool.acquire() as connection:
            # Определяем URL изображения (по умолчанию)
            final_image_url = "/static/images/product_default.jpg"

            # Обработка изображения
            if image_file and isinstance(image_file, UploadFile) and image_file.filename:
                print(f"📁 Загружаем файл: {image_file.filename}")
                # Проверяем расширение файла
                allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
                file_extension = Path(image_file.filename).suffix.lower()

                if file_extension not in allowed_extensions:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Недопустимый формат файла. Разрешены: {', '.join(allowed_extensions)}"
                    )

                # Генерируем уникальное имя файла
                filename = f"{uuid4().hex}{file_extension}"
                file_path = UPLOAD_DIR / filename

                # Сохраняем файл
                async with aiofiles.open(file_path, 'wb') as buffer:
                    content = await image_file.read()
                    await buffer.write(content)

                final_image_url = f"/static/uploads/{filename}"
                print(f"✅ Файл сохранен: {final_image_url}")

            elif image_url and image_url.strip():
                # Используем готовый URL если нет загружаемого файла
                final_image_url = image_url.strip()
                print(f"🔗 Используем URL изображения: {final_image_url}")
            else:
                print("🖼️ Используем изображение по умолчанию")

            # Создаем товар в базе данных
            print(f"💾 Сохраняем товар в базу: {name}, {category}, {price}")
            product = await connection.fetchrow('''
                INSERT INTO products (name, category, price, description, image_url, stock, featured)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
            ''', name, category, price, description, final_image_url, stock, featured)

            # Конвертируем Decimal в float
            product_dict = dict(product)
            product_dict['price'] = float(product['price'])

            print("✅ Товар успешно создан")

            return {
                "message": "Товар успешно создан",
                "product": product_dict
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка создания товара: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.put("/api/admin/products/{product_id}")
async def update_product(
        product_id: int,
        request: Request,
        admin=Depends(verify_admin)
):
    """Обновить товар - УНИВЕРСАЛЬНАЯ ВЕРСИЯ"""
    try:
        form_data = await request.form()

        # Извлекаем данные из формы
        name = form_data.get("name")
        category = form_data.get("category")
        price_str = form_data.get("price")
        description = form_data.get("description")
        stock_str = form_data.get("stock")
        featured_str = form_data.get("featured")
        image_url = form_data.get("image_url")

        async with db.pool.acquire() as connection:
            # Проверяем, существует ли товар
            product_exists = await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM products WHERE id = $1)",
                product_id
            )

            if not product_exists:
                raise HTTPException(status_code=404, detail="Товар не найден")

            # Получаем текущие данные товара
            current_product = await connection.fetchrow(
                "SELECT * FROM products WHERE id = $1",
                product_id
            )

            # Подготавливаем данные для обновления
            update_data = {}

            if name is not None:
                update_data['name'] = str(name).strip()
                if len(update_data['name']) < 3:
                    raise HTTPException(status_code=400, detail="Название товара должно содержать минимум 3 символа")
            else:
                update_data['name'] = current_product['name']

            if category is not None:
                update_data['category'] = str(category).strip()
            else:
                update_data['category'] = current_product['category']

            if price_str is not None:
                try:
                    update_data['price'] = float(price_str)
                    if update_data['price'] <= 0:
                        raise HTTPException(status_code=400, detail="Цена должна быть больше 0")
                except ValueError:
                    raise HTTPException(status_code=400, detail="Неверный формат цены")
            else:
                update_data['price'] = float(current_product['price'])

            if description is not None:
                update_data['description'] = str(description).strip()
                if len(update_data['description']) < 10:
                    raise HTTPException(status_code=400, detail="Описание должно содержать минимум 10 символов")
            else:
                update_data['description'] = current_product['description']

            if stock_str is not None:
                try:
                    update_data['stock'] = int(stock_str)
                    if update_data['stock'] < 0:
                        raise HTTPException(status_code=400, detail="Количество не может быть отрицательным")
                except ValueError:
                    raise HTTPException(status_code=400, detail="Неверный формат количества")
            else:
                update_data['stock'] = current_product['stock']

            if featured_str is not None:
                update_data['featured'] = featured_str.lower() == 'true'
            else:
                update_data['featured'] = current_product['featured']

            if image_url is not None:
                update_data['image_url'] = str(image_url).strip()
                if not update_data['image_url']:
                    update_data['image_url'] = "/static/images/product_default.jpg"
            else:
                update_data['image_url'] = current_product['image_url']

            # Выполняем обновление
            product = await connection.fetchrow('''
                UPDATE products 
                SET name = $1, category = $2, price = $3, description = $4, 
                    stock = $5, featured = $6, image_url = $7
                WHERE id = $8
                RETURNING *
            ''', update_data['name'], update_data['category'], update_data['price'],
                                                update_data['description'], update_data['stock'],
                                                update_data['featured'], update_data['image_url'], product_id)

            # Конвертируем Decimal в float
            product_dict = dict(product)
            product_dict['price'] = float(product['price'])

            return {
                "message": "Товар успешно обновлен",
                "product": product_dict
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка обновления товара: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.delete("/api/admin/products/{product_id}")
async def delete_product(
        product_id: int,
        admin=Depends(verify_admin)
):
    """Удалить товар"""
    try:
        async with db.pool.acquire() as connection:
            # Проверяем, существует ли товар
            product_exists = await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM products WHERE id = $1)",
                product_id
            )

            if not product_exists:
                raise HTTPException(status_code=404, detail="Товар не найден")

            # Получаем данные о товаре перед удалением
            product = await connection.fetchrow(
                "SELECT * FROM products WHERE id = $1",
                product_id
            )

            # Удаляем товар
            await connection.execute(
                "DELETE FROM products WHERE id = $1",
                product_id
            )

            # Конвертируем Decimal в float
            product_dict = dict(product)
            product_dict['price'] = float(product['price'])

            return {
                "message": "Товар успешно удален",
                "product": product_dict
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка удаления товара: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/admin/images")
async def get_available_images(admin=Depends(verify_admin)):
    """Получить список доступных изображений"""
    try:
        images = []

        # Сканируем папки с изображениями
        image_folders = [
            (STATIC_DIR / "images", "/static/images"),
            (UPLOAD_DIR, "/static/uploads")
        ]

        for folder, url_prefix in image_folders:
            if folder.exists():
                for file in folder.iterdir():
                    if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                        images.append({
                            "url": f"{url_prefix}/{file.name}",
                            "name": file.name,
                            "size": file.stat().st_size,
                            "path": str(file)
                        })

        return {"images": images}

    except Exception as e:
        print(f"Ошибка получения списка изображений: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.post("/api/admin/upload-image")
async def upload_image(
        image_file: UploadFile = File(...),
        admin=Depends(verify_admin)
):
    """Загрузить изображение"""
    try:
        if not image_file.filename:
            raise HTTPException(status_code=400, detail="Файл не выбран")

        # Проверяем расширение файла
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        file_extension = Path(image_file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Недопустимый формат файла. Разрешены: {', '.join(allowed_extensions)}"
            )

        # Проверяем размер файла (максимум 5MB)
        content = await image_file.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Размер файла не должен превышать 5MB")

        # Генерируем уникальное имя файла
        filename = f"{uuid4().hex}{file_extension}"
        file_path = UPLOAD_DIR / filename

        # Сохраняем файл
        async with aiofiles.open(file_path, 'wb') as buffer:
            await buffer.write(content)

        return {
            "message": "Изображение успешно загружено",
            "url": f"/static/uploads/{filename}",
            "filename": filename
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


# Тестовый эндпоинт для проверки
@app.get("/api/test-auth")
async def test_auth():
    """Тестирование аутентификации"""
    try:
        test_password = "test123"
        hashed = hasher.get_password_hash(test_password)
        verified = hasher.verify_password(test_password, hashed)

        async with db.pool.acquire() as connection:
            users_count = await connection.fetchval("SELECT COUNT(*) FROM users")
            demo_user_exists = await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM users WHERE username = 'demo')"
            )

        return {
            "status": "ok",
            "hash_working": verified,
            "users_count": users_count,
            "demo_user_exists": demo_user_exists,
            "admin_password_set": ADMIN_PASSWORD != "admin123",
            "database_connected": db.pool is not None
        }

    except Exception as e:
        print(f"Ошибка тестирования: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# ========== ГЛАВНАЯ СТРАНИЦА ==========
@app.get("/")
async def root(request: Request):
    """Главная страница магазина"""

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ========== СТРАНИЦА ВСЕХ ТОВАРОВ ==========
@app.get("/products")
async def products_page(request: Request):
    """Страница со всеми товарами"""

    return templates.TemplateResponse(
        "products.html",
        {"request": request}
    )


# ========== АДМИНСКАЯ ПАНЕЛЬ ==========
@app.get("/admin")
async def admin_panel(request: Request):
    """Админская панель"""

    return templates.TemplateResponse(
        "admin.html",
        {"request": request}
    )


# ========== ОТДЕЛЬНАЯ СТРАНИЦА ДЛЯ ДОБАВЛЕНИЯ ТОВАРА ==========
@app.get("/admin/add-product")
async def admin_add_product_page(request: Request):
    """Страница добавления товара"""
    return templates.TemplateResponse(
        "add_product.html",
        {"request": request}
    )


@app.post("/api/admin/add-product")
async def admin_add_product_api(
        request: Request,
        admin=Depends(verify_admin)
):
    """API для добавления товара (отдельный эндпоинт)"""
    try:
        form_data = await request.form()

        # Отладочная информация
        print("📥 [ADD-PRODUCT API] Получены данные формы:")
        for key, value in form_data.items():
            if key != 'image_file':
                print(f"  {key}: {value}")

        # Извлекаем данные из формы
        name = form_data.get("name")
        category = form_data.get("category")
        price_str = form_data.get("price")
        description = form_data.get("description")
        stock_str = form_data.get("stock")
        featured_str = form_data.get("featured")
        image_url = form_data.get("image_url")
        image_file = form_data.get("image_file")

        # Преобразуем типы
        if not name:
            raise HTTPException(status_code=400, detail="Название товара обязательно")
        name = str(name).strip()

        if not category:
            raise HTTPException(status_code=400, detail="Категория товара обязательна")
        category = str(category).strip()

        if not price_str:
            raise HTTPException(status_code=400, detail="Цена товара обязательна")
        try:
            price = float(price_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Неверный формат цены. Используйте числа")

        if not description:
            raise HTTPException(status_code=400, detail="Описание товара обязательно")
        description = str(description).strip()

        if not stock_str:
            stock = 0
        else:
            try:
                stock = int(stock_str)
            except ValueError:
                raise HTTPException(status_code=400, detail="Неверный формат количества")

        if not featured_str:
            featured = False
        else:
            featured = featured_str.lower() == 'true'

        if image_url:
            image_url = str(image_url).strip()

        # Валидация данных
        if len(name) < 3:
            raise HTTPException(
                status_code=400,
                detail="Название товара должно содержать минимум 3 символа"
            )

        if price <= 0:
            raise HTTPException(
                status_code=400,
                detail="Цена должна быть больше 0"
            )

        if len(description) < 10:
            raise HTTPException(
                status_code=400,
                detail="Описание должно содержать минимум 10 символов"
            )

        if stock < 0:
            raise HTTPException(
                status_code=400,
                detail="Количество не может быть отрицательным"
            )

        async with db.pool.acquire() as connection:
            # Определяем URL изображения (по умолчанию)
            final_image_url = "/static/images/product_default.jpg"

            # Обработка изображения
            if image_file and isinstance(image_file, UploadFile) and image_file.filename:
                print(f"📁 [ADD-PRODUCT] Загружаем файл: {image_file.filename}")
                # Проверяем расширение файла
                allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
                file_extension = Path(image_file.filename).suffix.lower()

                if file_extension not in allowed_extensions:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Недопустимый формат файла. Разрешены: {', '.join(allowed_extensions)}"
                    )

                # Генерируем уникальное имя файла
                filename = f"{uuid4().hex}{file_extension}"
                file_path = UPLOAD_DIR / filename

                # Сохраняем файл
                async with aiofiles.open(file_path, 'wb') as buffer:
                    content = await image_file.read()
                    await buffer.write(content)

                final_image_url = f"/static/uploads/{filename}"
                print(f"✅ [ADD-PRODUCT] Файл сохранен: {final_image_url}")

            elif image_url and image_url.strip():
                # Используем готовый URL если нет загружаемого файла
                final_image_url = image_url.strip()
                print(f"🔗 [ADD-PRODUCT] Используем URL изображения: {final_image_url}")
            else:
                print("🖼️ [ADD-PRODUCT] Используем изображение по умолчанию")

            # Создаем товар в базе данных
            print(f"💾 [ADD-PRODUCT] Сохраняем товар в базу: {name}, {category}, {price}")
            product = await connection.fetchrow('''
                INSERT INTO products (name, category, price, description, image_url, stock, featured)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
            ''', name, category, price, description, final_image_url, stock, featured)

            # Конвертируем Decimal в float
            product_dict = dict(product)
            product_dict['price'] = float(product['price'])

            print("✅ [ADD-PRODUCT] Товар успешно создан")

            return {
                "success": True,
                "message": "Товар успешно добавлен!",
                "product": product_dict
            }

    except HTTPException as he:
        print(f"❌ [ADD-PRODUCT] HTTPException: {he.detail}")
        raise
    except Exception as e:
        print(f"❌ [ADD-PRODUCT] Ошибка создания товара: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

import os
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ========== ЗАПУСК ==========
if __name__ == "__main__":
    print("=" * 70)
    print("🛴 Scooter Parts Shop v4.5")
    print("=" * 70)

    print("🌐 Сервер запущен:")
    print("   • http://localhost:8000              - Главная страница")
    print("   • http://localhost:8000/products     - Все товары")
    print("   • http://localhost:8000/admin        - Админка")
    print("   • http://localhost:8000/api/test-auth - Тест аутентификации")
    print("=" * 70)
    print("⚠️  Важно! Создайте файл .env с переменными:")
    print("   ADMIN_PASSWORD=ваш_пароль_для_админки")
    print("   SECRET_KEY=ваш_секретный_ключ_для_jwt")
    print("   DATABASE_URL=postgresql://user:password@localhost/scooter_shop")
    print("=" * 70)
    print("🔄 Для замены иконок:")
    print("   1. Подготовьте изображения в папке /static/favicon/")
    print("   2. Основные файлы: favicon.ico, favicon-32x32.png")
    print("   3. Очистите кэш браузера после замены (Ctrl+F5)")
    print("=" * 70)
    print("📦 Для добавления товаров в админке:")
    print("   1. Войдите в админку (admin + пароль)")
    print("   2. Перейдите в раздел 'Добавить товар'")
    print("   3. Заполните все обязательные поля")
    print("   4. Выберите или загрузите изображение")
    print("   5. Нажмите 'Сохранить товар'")
    print("=" * 70)

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)