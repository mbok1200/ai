import json
import requests, os
from typing import Dict
from datetime import datetime, timedelta

class RedmineAPI:
    """Клас для роботи з Redmine API"""
    
    def __init__(self):
        self.base_url = os.getenv("REDMINE_URL", "").rstrip('/')
        self.api_key = os.getenv("REDMINE_API_KEY")
        self.user_id = os.getenv("REDMINE_USER_ID")
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Базовий метод для HTTP запитів до Redmine"""
        if not self.base_url or not self.api_key:
            raise Exception("Redmine API не налаштований")
        
        url = f"{self.base_url}/issues.json"
        headers = {
            'X-Redmine-API-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=data)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data)
            elif method == "PUT":
                response = requests.put(url, headers=headers, json=data)
            else:
                raise ValueError(f"Непідтримуваний HTTP метод: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Помилка Redmine API: {str(e)}")
 
    def access_to_redmine(self) -> str:
        """Перевірка доступу до Redmine API"""
        try:
            url = f"{self.base_url}/issues.json"
            headers = {'X-Redmine-API-Key': self.api_key}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return "✅ Доступ до Redmine API підтверджено"
        except Exception as e:
            return f"❌ Помилка доступу до Redmine API: {str(e)}"

    def get_issue_by_id(self, issue_id: str) -> str:
        """Отримання завдання за ID"""
        try:
            # Очищуємо ID від # якщо є
            clean_id = issue_id.replace('#', '').strip()
            
            url = f"{self.base_url}/issues/{clean_id}.json"
            headers = {'X-Redmine-API-Key': self.api_key}
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            issue = response.json()['issue']
            
            return self._format_issue(issue)
            
        except Exception as e:
            return f"❌ Не вдалося знайти завдання {issue_id}: {str(e)}"
    
    def get_issue_by_date(self, date: str) -> str:
        """Отримання завдань за датою"""
        try:
            # Парсимо дату
            parsed_date = self._parse_date(date)
            
            params = {
                'assigned_to_id': self.user_id,
                'updated_on': f">={parsed_date}",
                'limit': 10
            }
            
            data = self._make_request('', params=params)
            
            if not data.get('issues'):
                return f"📅 На {date} завдань не знайдено"
            
            issues_text = [self._format_issue_short(issue) for issue in data['issues']]
            return f"📅 Завдання на {date}:\n\n" + "\n".join(issues_text)
            
        except Exception as e:
            return f"❌ Помилка пошуку завдань за датою: {str(e)}"
    
    def search_issues(self, search_term: str) -> str:
        """Пошук завдань за текстом"""
        try:
            params = {
                'assigned_to_id': self.user_id,
                'subject': f"~{search_term}",
                'limit': 5
            }
            
            data = self._make_request('', params=params)
            
            if not data.get('issues'):
                return f"🔍 За запитом '{search_term}' нічого не знайдено"
            
            issues_text = [self._format_issue_short(issue) for issue in data['issues']]
            return f"🔍 Результати пошуку '{search_term}':\n\n" + "\n".join(issues_text)
            
        except Exception as e:
            return f"❌ Помилка пошуку: {str(e)}"

    def get_issue_by_name(self, issue_name: str) -> str:
        """Отримання завдання за назвою"""
        try:
            params = {
                'assigned_to_id': self.user_id,
                'status_id': 'open',
                'subject': f"~{issue_name}",
                'limit': 5
            }

            data = self._make_request('', params=params)

            if not data.get('issues'):
                return f"🔍 За запитом '{issue_name}' нічого не знайдено"

            issues_text = [self._format_issue_short(issue) for issue in data['issues']]
            return f"🔍 Результати пошуку '{issue_name}':\n\n" + "\n".join(issues_text)

        except Exception as e:
            return f"❌ Помилка пошуку: {str(e)}"
    def get_issue_hours(self, issue_name: str) -> str:
        """Отримання годин по завданню"""
        try:
            params = {
                'assigned_to_id': self.user_id,
                'subject': f"~{issue_name}",
                'limit': 1
            }
            
            data = self._make_request('', params=params)
            
            if not data.get('issues'):
                return f"🔍 За запитом '{issue_name}' нічого не знайдено"
            
            issue = data['issues'][0]
            hours = issue.get('estimated_hours', 0)
            return f"⏱️ Години по завданню '{issue_name}': {hours} год."
            
        except Exception as e:
            return f"❌ Помилка отримання годин: {str(e)}"
    def fill_issue_hours(self, issue_id: str, hours: float, description: str = "") -> str:
        """Заповнення годин по завданню"""
        try:
            clean_id = issue_id.replace('#', '').strip()
            
            url = f"{self.base_url}/issues/{clean_id}.json"
            headers = {
                'X-Redmine-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'issue': {
                    'estimated_hours': hours,
                    'notes': description
                }
            }
            
            response = requests.put(url, headers=headers, json=data)
            response.raise_for_status()
            
            return f"✅ Заповнено {hours} год. для завдання #{clean_id}"
            
        except Exception as e:
            return f"❌ Помилка заповнення годин: {str(e)}"
    def get_user_status(self) -> str:
        """Отримання статусу користувача"""
        try:
            url = f"{self.base_url}/users/{self.user_id}.json"
            headers = {'X-Redmine-API-Key': self.api_key}
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            user = response.json()['user']
            status = user.get('status', 'Невідомо')
            
            return f"👤 Статус користувача: {status}"
            
        except Exception as e:
            return f"❌ Помилка отримання статусу користувача: {str(e)}"
    def set_user_status(self, status: str) -> str:
        """Встановлення статусу користувача"""
        try:
            url = f"{self.base_url}/users/{self.user_id}.json"
            headers = {
                'X-Redmine-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'user': {
                    'status': status
                }
            }
            
            response = requests.put(url, headers=headers, json=data)
            response.raise_for_status()
            
            return f"✅ Статус користувача змінено на: {status}"
            
        except Exception as e:
            return f"❌ Помилка встановлення статусу: {str(e)}"
    def create_issue(self, subject: str, description: str = "", priority: str = "Normal") -> str:
        """Створення нового завдання"""
        try:
            url = f"{self.base_url}/issues.json"
            headers = {
                'X-Redmine-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'issue': {
                    'subject': subject,
                    'description': description,
                    'priority_id': self._get_priority_id(priority)
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            issue = response.json()['issue']
            return f"✅ Завдання створено: {self._format_issue(issue)}"
            
        except Exception as e:
            return f"❌ Помилка створення завдання: {str(e)}"
    def assign_issue(self, issue_id: str, user_id: str) -> str: 
        """Призначення завдання користувачу"""
        try:
            clean_id = issue_id.replace('#', '').strip()
            
            url = f"{self.base_url}/issues/{clean_id}.json"
            headers = {
                'X-Redmine-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'issue': {
                    'assigned_to_id': user_id
                }
            }
            
            response = requests.put(url, headers=headers, json=data)
            response.raise_for_status()
            
            return f"✅ Завдання #{clean_id} призначено користувачу {user_id}"
            
        except Exception as e:
            return f"❌ Помилка призначення завдання: {str(e)}"
    def get_wiki_info(self, topic: str) -> str:
        """Отримання інформації з Wiki"""
        try:
            url = f"{self.base_url}/wiki/{topic}.json"
            headers = {'X-Redmine-API-Key': self.api_key}
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            wiki_info = response.json()['wiki']
            return f"📖 Wiki інформація про {topic}: {wiki_info['content'][:200]}..."
            
        except Exception as e:
            return f"❌ Помилка отримання Wiki інформації: {str(e)}"
    
    def _format_issue(self, issue: Dict) -> str:
        """Форматування повної інформації про завдання"""
        title = issue.get('subject', 'Без назви')
        status = issue.get('status', {}).get('name', 'Невідомо')
        priority = issue.get('priority', {}).get('name', 'Невідомо')
        assignee = issue.get('assigned_to', {}).get('name', 'Не призначено')
        description = issue.get('description', '')[:200] + '...' if issue.get('description') else ''
        
        return f"""🎯 **Завдання #{issue['id']}**
📝 **Назва:** {title}
📊 **Статус:** {status}
⚡ **Пріоритет:** {priority}
👤 **Відповідальний:** {assignee}
📄 **Опис:** {description}"""
    
    def _format_issue_short(self, issue: Dict) -> str:
        """Короткий формат завдання"""
        title = issue.get('subject', 'Без назви')
        status = issue.get('status', {}).get('name', 'Невідомо')
        
        return f"#{issue['id']} - {title} ({status})"
    
    def _parse_date(self, date_str: str) -> str:
        """Парсинг дати в формат Redmine"""
        date_str = date_str.lower().strip()
        
        today = datetime.now()
        
        if date_str in ['сьогодні', 'today']:
            return today.strftime('%Y-%m-%d')
        elif date_str in ['вчора', 'yesterday']:
            return (today - timedelta(days=1)).strftime('%Y-%m-%d')
        elif date_str in ['завтра', 'tomorrow']:
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            # Спробуємо парсити як дату
            try:
                # Формат дд.мм.рррр або дд.мм
                if '.' in date_str:
                    parts = date_str.split('.')
                    if len(parts) == 2:
                        day, month = int(parts[0]), int(parts[1])
                        year = today.year
                        return f"{year:04d}-{month:02d}-{day:02d}"
                    elif len(parts) == 3:
                        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                        if year < 100:
                            year += 2000
                        return f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                pass
        
        # Якщо не вдалося парсити, повертаємо сьогодні
        return today.strftime('%Y-%m-%d')