import os
import json
from twilio.rest import Client

class NotificationService:

    def send_otp(self, to_number, otp_code):
        """
        Sends a 6-digit OTP code.
        """
        if not self.client: return False
        
        # Clean number
        clean_number = to_number.strip()
        if not clean_number.startswith('+'): clean_number = f"+91{clean_number}"

        msg_body = f"🔐 Your AgriAssist Login OTP is: *{otp_code}*\nDo not share this with anyone."

        try:
            message = self.client.messages.create(
                body=msg_body,
                from_=self.from_whatsapp,
                to=f"whatsapp:{clean_number}"
            )
            print(f"✅ OTP Sent: {otp_code}")
            return True
        except Exception as e:
            print(f"❌ OTP Failed: {e}")
            return False
        

    def __init__(self):
        # 1. Credentials
        self.account_sid = "AC677aaad61269a1079aeddaba8f2a428a" 
        self.auth_token = "ea2a9be5f3bc5f78518ec2f55de1b37a"
        
        # 2. Config
        self.from_whatsapp = "whatsapp:+14155238886" 
        
        # ⚠️ YOUR NEW TEMPLATE ID (irrigationalert)
        self.content_sid = "HX4a3c0e2cfad2ae05fca3dfa261287bdf"

        try:
            self.client = Client(self.account_sid, self.auth_token)
        except:
            self.client = None

    def send_irrigation_alert(self, to_number, var_1, var_2):
        """
        Sends the Template Message.
        var_1: e.g. Crop Name ({{1}})
        var_2: e.g. Date ({{2}})
        """
        if not self.client:
            print("❌ Twilio client not ready.")
            return False

        # Format number to +91...
        clean_number = to_number.strip()
        if not clean_number.startswith('+'):
            clean_number = f"+91{clean_number}"
        
        # Define the variables to plug into the template
        # This matches the 'ContentVariables' in your curl command
        variables = json.dumps({
            "1": str(var_1), 
            "2": str(var_2)
        })

        try:
            message = self.client.messages.create(
                from_=self.from_whatsapp,
                to=f"whatsapp:{clean_number}",
                content_sid=self.content_sid,
                content_variables=variables
            )
            print(f"✅ Template Sent! SID: {message.sid}")
            return True
        except Exception as e:
            print(f"❌ Twilio Error: {e}")
            return False